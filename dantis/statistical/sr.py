"""
Spectral Residual detector adapted to `AlgorithmBase`.

Esta implementación copia/adapta la lógica principal del Spectral Residual de la librería 

Clase principal: `SpectralResidual` (implementa `fit`,
`decision_function`, `predict`, `get_default_hyperparameters`, `set_hyperparameter`).
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
import pandas as pd

from .. import algorithmbase
from .. import utils
from .sr_utils import (
    deanomaly_entire,
    median_filter,
    calculate_boundary_unit_entire,
    calculate_expected_value,
    calculate_anomaly_scores,
)


# --- Helper constants and functions (self-contained) ---
EPS = 1e-8


def average_filter(values: np.ndarray, n: int = 3) -> np.ndarray:
    if n >= len(values):
        n = len(values)

    res = np.cumsum(values, dtype=float)
    res[n:] = res[n:] - res[:-n]
    res[n:] = res[n:] / n

    for i in range(1, n):
        res[i] /= (i + 1)

    return res


def predict_next(values: np.ndarray) -> float:
    if len(values) <= 1:
        raise ValueError('data should contain at least 2 numbers')
    v_last = values[-1]
    n = len(values)
    slopes = [(v_last - v) / (n - 1 - i) for i, v in enumerate(values[:-1])]
    return values[1] + sum(slopes)


def extend_series(values: np.ndarray, extend_num: int = 5, look_ahead: int = 5) -> np.ndarray:
    if look_ahead < 1:
        raise ValueError('look_ahead must be at least 1')
    extension = [predict_next(values[-look_ahead - 2:-1])] * extend_num
    return np.concatenate((values, extension), axis=0)


def spectral_residual_transform(values: np.ndarray, mag_window: int) -> np.ndarray:
    trans = np.fft.fft(values)
    mag = np.sqrt(trans.real ** 2 + trans.imag ** 2)
    eps_index = np.where(mag <= EPS)[0]
    mag[eps_index] = EPS

    mag_log = np.log(mag)
    mag_log[eps_index] = 0

    spectral = np.exp(mag_log - average_filter(mag_log, n=mag_window))

    trans.real = trans.real * spectral / mag
    trans.imag = trans.imag * spectral / mag
    trans.real[eps_index] = 0
    trans.imag[eps_index] = 0

    wave_r = np.fft.ifft(trans)
    mag = np.sqrt(wave_r.real ** 2 + wave_r.imag ** 2)
    return mag


def generate_spectral_score(mags: np.ndarray, score_window: int) -> np.ndarray:
    ave_mag = average_filter(mags, n=score_window)
    safeDivisors = np.clip(ave_mag, EPS, ave_mag.max() if ave_mag.size else EPS)
    raw_scores = np.abs(mags - ave_mag) / safeDivisors
    scores = np.clip(raw_scores / 10.0, 0, 1.0)
    return scores


# --- Wrapper class ---
class SpectralResidual(algorithmbase.AlgorithmBase):
    """Spectral Residual detector wrapper.

    Input accepted: 1-D numpy array/sequence or 2-col DataFrame (`timestamp`,`value`).
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "threshold": 0.3,
            "mag_window": 3,
            "score_window": 40,
            "sensitivity": 0.5,
            "detect_mode": "anomaly_only",
            "batch_size": 0,
            "contamination": 0.1,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._last_scores = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults.copy()
            return

        def _hp_template(threshold=0.3, mag_window=3, score_window=40,
                         sensitivity=0.5, detect_mode="anomaly_only", batch_size=0,
                         contamination=0.1):
            return None

        pm = utils.ParameterManagement(_hp_template)
        try:
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"Invalid hyperparameter types: {e}")

        merged = pm.complete_parameters(coerced)
        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v

        self._hyperparameter = merged

    def _to_dataframe(self, x) -> pd.DataFrame:
        if isinstance(x, pd.DataFrame):
            df = x.copy()
            if df.shape[1] >= 2:
                df = df.iloc[:, :2]
            df.columns = ["timestamp", "value"]
            return df

        arr = np.asarray(x)
        if arr.ndim == 1:
            return pd.DataFrame({"timestamp": np.arange(len(arr)), "value": arr.astype(float)})
        elif arr.ndim == 2 and arr.shape[1] >= 2:
            return pd.DataFrame({"timestamp": arr[:, 0], "value": arr[:, 1].astype(float)})
        else:
            raise ValueError("Unsupported input shape for SpectralResidual")

    def _compute_scores_for_df(self, df: pd.DataFrame) -> np.ndarray:
        values = df['value'].values.astype(float)
        params = self._hyperparameter
        mag_window = int(params.get('mag_window', 3))
        score_window = int(params.get('score_window', 40))
        batch_size = int(params.get('batch_size', 0))

        if batch_size <= 0:
            batch_size = len(values)
        batch_size = max(12, batch_size)
        batch_size = min(len(values), batch_size)

        scores_all = np.zeros(len(values))
        for i in range(0, len(values), batch_size):
            start = i
            end = min(i + batch_size, len(values))
            if end - start < 12:
                ext_start = max(0, end - batch_size)
                window_vals = values[ext_start:end]
                ext_offset = start - ext_start
            else:
                window_vals = values[start:end]
                ext_offset = 0
            extended = extend_series(window_vals)
            mags = spectral_residual_transform(extended, mag_window=mag_window)
            anomaly_scores = generate_spectral_score(mags, score_window=score_window)
            usable = anomaly_scores[:len(window_vals)]

            # If anomaly_and_margin mode, compute expected values, units and
            # recompute anomaly scores using margin logic to obtain final scores
            if str(self._hyperparameter.get('detect_mode', 'anomaly_only')) == 'anomaly_and_margin':
                thresh = float(self._hyperparameter.get('threshold', 0.3))
                is_anom = (usable > thresh)
                # anomaly indices relative to window
                anom_indices = [int(idx) for idx, flag in enumerate(is_anom) if flag]
                expected = calculate_expected_value(window_vals, anom_indices)
                units = calculate_boundary_unit_entire(window_vals, is_anom)
                # recompute final scores for the whole window
                final_scores = np.array(calculate_anomaly_scores(window_vals, expected, units, is_anom), dtype=float)
                usable = final_scores[:len(window_vals)]
            scores_all[start:end] = usable[ext_offset:ext_offset + (end - start)]

        return scores_all

    def fit(self, x_train, y_train=None):
        df = self._to_dataframe(x_train)
        scores = self._compute_scores_for_df(df)
        self._last_scores = scores
        return scores

    def decision_function(self, x):
        df = self._to_dataframe(x)
        return self._compute_scores_for_df(df)

    def predict(self, x):
        scores = self.get_anomaly_score(x)
        params = self._hyperparameter
        # Use explicit threshold if provided, otherwise contamination percentile
        threshold = params.get('threshold', None)
        if threshold is not None:
            thr = float(threshold)
            labels = (scores > thr).astype(int)
            return labels

        contamination = float(params.get('contamination', 0.1))
        contamination = max(0.001, min(0.5, contamination))
        valid = np.isfinite(scores)
        if not np.any(valid):
            return np.zeros_like(scores, dtype=int)
        thr = np.percentile(scores[valid], 100.0 * (1.0 - contamination))
        labels = np.zeros_like(scores, dtype=int)
        labels[valid] = (scores[valid] >= thr).astype(int)
        return labels
