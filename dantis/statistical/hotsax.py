import numpy as np
from collections import defaultdict, Counter
from typing import Optional, Dict, Any, List, Tuple

from .. import algorithmbase
from .. import utils
"""
HOTSAX wrapper for dantis using saxpy as dependency.

This module delegates the HOT-SAX logic to the `saxpy` package and provides
an `HOTSAX` wrapper compatible with `AlgorithmBase`.
"""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, Any

from saxpy.hotsax import find_discords_hotsax

from .. import algorithmbase


class HOTSAX(algorithmbase.AlgorithmBase):
    """Wrapper for HOT-SAX discord discovery using saxpy.

    Hyperparameters:
    - anomaly_window_size (win_size)
    - paa_transform_size (paa_size)
    - alphabet_size
    - normalization_threshold
    - random_state
    - num_discords
    - contamination (for `predict` conversion)
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "anomaly_window_size": 100,
            "paa_transform_size": 3,
            "alphabet_size": 3,
            "normalization_threshold": 0.01,
            "random_state": 42,
            "num_discords": None,
            "contamination": 0.1,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults
            return
        merged = defaults.copy()
        merged.update(hyperparameter)
        # Use ParameterManagement to validate/complete types based on defaults
        try:
            pm = utils.ParameterManagement(lambda: None)
            pm._default_parameters = defaults.copy()
            validated = pm.check_hyperparameter_type(merged)
            merged = pm.complete_parameters(validated)
        except Exception:
            # fall back to merged
            pass
        self._hyperparameter = merged

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        # HOTSAX is unsupervised and does not require a training phase.
        self.set_hyperparameter(self._hyperparameter)
        # return scores on training data for convenience
        return self.decision_function(x_train)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for a 1-D time series using saxpy.find_discords_hotsax.

        Returns an array of length len(x) with 0.0 for normal positions and
        a neighborhood-distance score for positions flagged as discords.
        """
        self.set_hyperparameter(self._hyperparameter)
        rs = self._hyperparameter.get("random_state", None)
        if rs is not None:
            import random

            random.seed(rs)
            np.random.seed(rs)

        series = np.asarray(x).flatten()
        win_size = int(self._hyperparameter.get("anomaly_window_size", 100))
        paa_size = int(self._hyperparameter.get("paa_transform_size", 3))
        alphabet_size = int(self._hyperparameter.get("alphabet_size", 3))
        znorm_threshold = float(self._hyperparameter.get("normalization_threshold", 0.01))
        num_discords = self._hyperparameter.get("num_discords", None)
        if num_discords is None:
            num_discords = max(1, len(series) - win_size + 1)

        if win_size < 1 or win_size > len(series):
            raise ValueError("anomaly_window_size must be between 1 and len(series)")
        if paa_size > win_size:
            paa_size = win_size

        discords = find_discords_hotsax(
            series,
            win_size=win_size,
            num_discords=num_discords,
            alphabet_size=alphabet_size,
            paa_size=paa_size,
            znorm_threshold=znorm_threshold,
            sax_type="unidim",
        )

        scores = np.zeros(len(series), dtype=float)
        for pos, score in discords:
            if 0 <= pos < len(scores):
                scores[pos] = score
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.get_anomaly_score(x)
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        if not (0 < contamination < 0.5):
            contamination = max(0.001, min(0.5, contamination))
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        return (scores >= threshold).astype(int)

    def save_model(self, path_model: str):
        # No persistent model to save for HOTSAX; keep interface behaviour
        super().save_model(path_model)

    def load_model(self, path_model: str):
        super().load_model(path_model)
