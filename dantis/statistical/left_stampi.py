from typing import Optional, Dict, Any

import sys
import numpy as np
import pandas as pd

from .. import algorithmbase
from .. import utils

try:
    from stumpy import stumpi
except Exception:
    stumpi = None


class LeftStampi(algorithmbase.AlgorithmBase):
    """Wrapper for the Left-STAMPI streaming matrix profile from stumpy.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "anomaly_window_size": 50,
            "n_init_train": 100,
            "random_state": 42,
            "use_column_index": 0,
            "contamination": 0.1,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._stream = None
        self._last_scores: Optional[np.ndarray] = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults
            return
        merged = defaults.copy()
        merged.update(hyperparameter)
        try:
            pm = utils.ParameterManagement(lambda **kwargs: None)
            pm._default_parameters = defaults.copy()
            validated = pm.check_hyperparameter_type(merged)
            merged = pm.complete_parameters(validated)
        except Exception:
            pass
        self._hyperparameter = merged

    def _create_stream(self, data: np.ndarray):
        if stumpi is None:
            raise RuntimeError("stumpy.stumpi is not available in the current environment")
        hp = self._hyperparameter
        warmup = int(hp.get("n_init_train", 100))
        ws = int(hp.get("anomaly_window_size", 50))
        if ws > warmup:
            ws = warmup
        if ws < 3:
            ws = 3
        # initialize streaming matrix profile
        return stumpi(data[:warmup], m=ws, egress=False)

    def _extract_series(self, x: Any) -> np.ndarray:
        """Return a 1-D float numpy array from a DataFrame or ndarray.

        If `x` is a DataFrame, respect `use_column_index` and skip the
        timestamp/index column.
        """
        # DataFrame handling 
        if isinstance(x, pd.DataFrame):
            df = x
            column_index = 0
            uci = self._hyperparameter.get("use_column_index", 0)
            if uci is not None:
                try:
                    column_index = int(uci)
                except Exception:
                    column_index = 0
            max_column_index = max(0, df.shape[1] - 3)
            if column_index > max_column_index:
                print(
                    f"Selected column index {column_index} is out of bounds (columns = {df.columns.values}; "
                    f"max index = {max_column_index}); using last channel!",
                    file=sys.stderr,
                )
                column_index = max_column_index
            # jump over index/timestamp column
            column_index = column_index + 1
            return df.values[:, column_index].astype(float)

        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr.astype(float)
        if arr.ndim == 2:
            # choose column based on use_column_index (no timestamp column assumed)
            column_index = int(self._hyperparameter.get("use_column_index", 0))
            if column_index >= arr.shape[1]:
                column_index = arr.shape[1] - 1
            return arr[:, column_index].astype(float)
        # fallback
        return arr.ravel().astype(float)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """Run the streaming left STAMPI on `x_train` and return point-wise scores.

        This follows the original script: build a stream from the first
        `n_init_train` points and then update it with the remaining points.
        The resulting `left_P_` is returned, with the warmup region zeroed.
        """
        self.set_hyperparameter(self._hyperparameter)
        # accept DataFrame or ndarray; extract the univariate series
        X_series = self._extract_series(x_train)
        X = np.asarray(X_series).astype(float)
        hp = self._hyperparameter
        warmup = int(hp.get("n_init_train", 100))
        if warmup >= X.shape[0]:
            # nothing to stream; compute left_P_ as zeros
            scores = np.zeros(X.shape[0], dtype=float)
            self._last_scores = scores
            return scores

        stream = self._create_stream(X)
        # feed remaining points
        for point in X[warmup:]:
            stream.update(point)

        mp = stream.left_P_
        # mask warmup region as in original script
        mp[:warmup] = 0.0
        self._stream = stream
        self._last_scores = np.asarray(mp)
        return self._last_scores

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        # If the model was already run and the input length (after
        # extraction) matches, return cached scores.
        series = self._extract_series(x)
        if self._last_scores is not None and len(self._last_scores) == len(series):
            return self._last_scores
        # otherwise recompute by running fit on the extracted series
        return self.fit(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.get_anomaly_score(x)
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        if not (0 < contamination < 0.5):
            contamination = max(0.001, min(0.5, contamination))
        valid = np.isfinite(scores)
        if not np.any(valid):
            return np.zeros_like(scores, dtype=int)
        threshold = np.percentile(scores[valid], 100.0 * (1.0 - contamination))
        labels = np.zeros_like(scores, dtype=int)
        labels[valid] = (scores[valid] >= threshold).astype(int)
        return labels

    def save_model(self, path_model: str):
        super().save_model(path_model)

    def load_model(self, path_model: str):
        super().load_model(path_model)
        # loaded object should restore _last_scores and _stream if present
