from pathlib import Path
from typing import Optional

import numpy as np

from ..algorithmbase import AlgorithmBase

# try to import the real eif package, fall back to local shim
try:
    import eif as _eif
except Exception:
    # local relative import
    from . import eif as _eif


class EIFWrapper(AlgorithmBase):
    """Wrapper que adapta la biblioteca `eif` al contrato de `AlgorithmBase`.

    Hyperparameters esperados en `hyperparameter`:
    - n_trees: int
    - max_samples: Optional[float] or int
    - extension_level: Optional[int]
    - limit: Optional[int]
    - threshold: Optional[float] -> umbral para predict sobre scores normalizados
    - random_state: int
    """

    def __init__(self, hyperparameter: dict | None = None) -> None:
        super().__init__(hyperparameter or {})
        self._forest = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        hp = self._hyperparameter or {}
        ntrees = int(hp.get("n_trees", 200))
        max_samples = hp.get("max_samples", None)
        limit = hp.get("limit", None)
        extension_level = hp.get("extension_level", None)
        random_state = hp.get("random_state", None)

        X = np.asarray(x_train)
        # remove potential 1D inputs
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # If max_samples is a fraction, convert to int sample_size later inside iForest
        sample_size = None
        if max_samples is not None and isinstance(max_samples, float):
            sample_size = max(1, int(max_samples * X.shape[0]))

        # Create the forest using the eif API
        # The iForest implementation expects X as first arg per the repo's shim
        self._forest = _eif.iForest(
            X,
            ntrees=ntrees,
            sample_size=sample_size or min(256, X.shape[0]),
            limit=limit,
            ExtensionLevel=extension_level,
            random_state=random_state,
        )
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self._forest is None:
            # lazy-fit: build forest from x
            self.fit(x)

        X = np.asarray(x)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        scores = self._forest.compute_paths(X_in=X)
        scores = np.asarray(scores, dtype=float)

        # Normalize to [0,1]
        if scores.max() > scores.min():
            scores = (scores - scores.min()) / (scores.max() - scores.min())
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        hp = self._hyperparameter or {}
        threshold = hp.get("threshold", None)
        if threshold is None:
            return (scores > 0).astype(int)
        return (scores >= float(threshold)).astype(int)
