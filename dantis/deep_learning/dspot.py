from dataclasses import dataclass
import numpy as np
from typing import Optional

from ..algorithmbase import AlgorithmBase

try:
    import pyspot as ps
except Exception:  # pragma: no cover - optional dependency
    ps = None


@dataclass
class CustomParameters:
    q: float = 1e-3
    n_init: int = 1000
    level: float = 0.99
    max_excess: int = 200
    up: bool = True
    down: bool = True
    alert: bool = True
    bounded: bool = True
    random_state: int = 42


class DSpot(AlgorithmBase):
    """Wrapper for `pyspot.Spot` that conforms to `AlgorithmBase` interface.

    Usage:
      model = DSpot({'customParameters': CustomParameters()})
      model.fit(x_train)
      scores = model.decision_function(x_test)
      preds = model.predict(x_test)
    """

    def __init__(self, hyperparameter: dict):
        super().__init__(hyperparameter)
        self.custom: CustomParameters = hyperparameter.get('customParameters', CustomParameters())
        self._spot: Optional[object] = None

    def _init_spot(self):
        if ps is None:
            raise ImportError("pyspot is required for DSpot but is not available")
        if self._spot is None:
            self._spot = ps.Spot(
                q=self.custom.q,
                n_init=self.custom.n_init,
                level=self.custom.level,
                max_excess=self.custom.max_excess,
                up=self.custom.up,
                down=self.custom.down,
                alert=self.custom.alert,
                bounded=self.custom.bounded,
            )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """Fit is a noop for Spot; we initialise the internal Spot object and
        run the initial batch (if any) by stepping through `x_train`.

        The original script didn't separate train/execute; to remain backward
        compatible we process `x_train` with `spot.step` so the internal models
        are warmed up.
        """
        assert isinstance(x_train, np.ndarray)
        # sanity check similar to original script
        assert self.custom.n_init * (1 - self.custom.level) > 10, (
            "too few data for calibration; either increase n_init or reduce level"
        )
        self._init_spot()
        # run through initial data to let spot calibrate/consume data
        for v in x_train:
            self._spot.step(float(v))
        # keep a reference for saving via AlgorithmBase
        self._model = self._spot

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Return anomaly score for each sample in `x`.

        The original Spot returns integer event codes. We map any alert
        (1 or -1) to score 1.0 and others to 0.0. If finer-grained scoring is
        needed, adapt this method accordingly.
        """
        if not isinstance(x, np.ndarray):
            x = np.asarray(x)
        self._init_spot()
        scores = np.zeros(x.shape[0], dtype=float)
        for i, v in enumerate(x):
            event = self._spot.step(float(v))
            if event in (1, -1):
                scores[i] = 1.0
            else:
                scores[i] = 0.0
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Binary predictions derived from `decision_function`.

        Returns 1 for anomaly, 0 for normal.
        """
        scores = self.decision_function(x)
        return (scores > 0).astype(int)
