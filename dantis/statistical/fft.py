"""Implementación del detector de anomalías basado en FFT.

Esta versión adapta la implementación de TimeEval (carpeta `fft`) a la
jerarquía `AlgorithmBase` definida en `dantis.algorithmbase`.

La clase principal es `FFTAlgorithm` que implementa `fit` (no-op) y
`decision_function` (devuelve puntuaciones de anomalía). También se
proporciona `predict` que devuelve etiquetas binarias simples.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import logging
import numpy as np

from dantis.algorithmbase import AlgorithmBase

logger = logging.getLogger(__name__)


@dataclass
class LocalOutlier:
    index: int
    z_score: float

    @property
    def sign(self) -> int:
        return int(np.sign(self.z_score))


@dataclass
class RegionOutlier:
    start_idx: int
    end_idx: int
    score: float


def reduce_parameters(f: np.ndarray, k: int) -> np.ndarray:
    transformed = f.copy()
    n = transformed.size
    if k <= 1:
        transformed[1:] = 0.0
    else:
        # keep k coefficients (including zero freq) symmetrically
        transformed[k:-(k - 1)] = 0
    return transformed


def calculate_local_outlier(data: np.ndarray, k: int, c: int, threshold: float) -> List[LocalOutlier]:
    n = len(data)
    k = max(min(k, n), 1)
    y = reduce_parameters(np.fft.fft(data), k)
    f2 = np.real(np.fft.ifft(y))
    so = np.abs(f2 - data)
    mso = np.mean(so)

    scores = []
    score_idxs = []
    for i in range(n):
        if so[i] > mso:
            nav = np.average(data[max(i - c, 0):min(i + c, n - 1) + 1])
            scores.append(data[i] - nav)
            score_idxs.append(i)

    if len(scores) == 0:
        return []

    scores = np.array(scores)
    ms = np.mean(scores)
    sds = np.std(scores)
    if sds == 0:
        return []

    results: List[LocalOutlier] = []
    for i in range(len(scores)):
        z_score = (scores[i] - ms) / sds
        # keep same semantics: compare abs(z_score) to threshold * sds (threshold in original code multiplies sigma)
        if abs(z_score) > threshold * sds:
            index = score_idxs[i]
            results.append(LocalOutlier(index, z_score))
    return results


def calculate_region_outlier(l_outliers: List[LocalOutlier], max_region_length: int, max_local_diff: int) -> List[RegionOutlier]:
    def distance(a: int, b: int) -> int:
        if a > b:
            a, b = b, a
        return l_outliers[b].index - l_outliers[a].index

    regions: List[RegionOutlier] = []
    i = 0
    n_l = len(l_outliers) - 1
    while i < n_l:
        s_sign = l_outliers[i].sign
        s_sign2 = l_outliers[i + 1].sign
        if s_sign != s_sign2 and distance(i, i + 1) <= max_local_diff:
            i += 1
            start_idx = i
            for j in range(i + 1, n_l + 1):
                # boundary-safe iteration
                if j >= n_l + 1:
                    break
                e_sign = l_outliers[j - 1].sign
                e_sign2 = l_outliers[j].sign
                if s_sign2 == e_sign and distance(start_idx, j - 1) <= max_region_length \
                        and e_sign != e_sign2 and distance(j - 1, j) <= max_local_diff:
                    end_idx = j - 1
                    regions.append(RegionOutlier(
                        start_idx=start_idx,
                        end_idx=end_idx,
                        score=float(np.mean([abs(l.z_score) for l in l_outliers[start_idx: end_idx + 1]]))
                    ))
                    i = j
                    break
            i = start_idx
        else:
            i += 1

    return regions


def detect_anomalies(data: np.ndarray,
                     ifft_parameters: int = 5,
                     local_neighbor_window: int = 21,
                     local_outlier_threshold: float = .6,
                     max_region_size: int = 50,
                     max_sign_change_distance: int = 10,
                     **_kwargs) -> np.ndarray:
    """Devuelve puntuaciones de anomalía (float) con la misma longitud que `data`.

    Parámetros adaptados desde la implementación original.
    """
    data = np.asarray(data).ravel()
    n = len(data)
    if n == 0:
        return np.array([])

    # neighbor window half-size
    neighbor_c = max(local_neighbor_window // 2, 1)

    local_outliers = calculate_local_outlier(data, ifft_parameters, neighbor_c, local_outlier_threshold)

    regions = calculate_region_outlier(local_outliers, max_region_size, max_sign_change_distance)

    anomaly_scores = np.zeros_like(data, dtype=float)
    for reg in regions:
        start_local = local_outliers[reg.start_idx]
        end_local = local_outliers[reg.end_idx]
        anomaly_scores[start_local.index:end_local.index + 1] = reg.score

    return anomaly_scores


class FFTAlgorithm(AlgorithmBase):
    """Detector basado en FFT.

    Hyperparameters admitidos (valores por defecto si no vienen en `hyperparameter`):
      - ifft_parameters: int = 5
      - context_window_size: int = 21
      - local_outlier_threshold: float = 0.6
      - max_anomaly_window_size: int = 50
      - max_sign_change_distance: int = 10

    Esta clase es no supervisada: `fit` es un no-op.
    """

    def __init__(self, hyperparameter: Optional[dict] = None) -> None:
        defaults = {
            "ifft_parameters": 5,
            "context_window_size": 21,
            "local_outlier_threshold": 0.6,
            "max_anomaly_window_size": 50,
            "max_sign_change_distance": 10,
            "random_state": 42,
        }
        merged = defaults.copy()
        if hyperparameter:
            merged.update(hyperparameter)
        super().__init__(merged)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """No hay entrenamiento para FFT; mantenemos la interfaz."""
        # nothing to train; keep model None
        self._model = None
        scores = self.decision_function(x_train)
        return (scores > 0).astype(int)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x)
        if x.size == 0:
            return np.array([])
        # if constant signal, no anomalies
        if np.allclose(x, x.flat[0]):
            return np.zeros_like(x, dtype=float)

        params = self.get_hyperparameter() or {}
        scores = detect_anomalies(
            x.ravel(),
            ifft_parameters=int(params.get("ifft_parameters", 5)),
            local_neighbor_window=int(params.get("context_window_size", 21)),
            local_outlier_threshold=float(params.get("local_outlier_threshold", 0.6)),
            max_region_size=int(params.get("max_anomaly_window_size", 50)),
            max_sign_change_distance=int(params.get("max_sign_change_distance", 10)),
        )

        # return with same shape as input flattened to 1D (caller may reshape)
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Predicción binaria simple: 1 si la puntuación > 0, 0 en caso contrario."""
        scores = self.decision_function(x)
        return (scores > 0).astype(int)
