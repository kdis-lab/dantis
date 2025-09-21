from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from .. import algorithmbase
from .. import utils

try:
    import pyspot as ps
except Exception:
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


class DSpot(algorithmbase.AlgorithmBase):
    """Adapter for the pylibspot based DSpot algorithm.

    Parameters
    ----------
    hyperparameter : dict
        Passed to the underlying pyspot Spot constructor.
    """

    def __init__(self, hyperparameter: Dict[str, Any] = None):
        # use a simple ParameterManagement based on the example function signature
        pm = utils.ParameterManagement(lambda q=1e-3, n_init=1000, level=0.99, max_excess=200, up=True, down=True, alert=True, bounded=True, random_state=42: None)
        defaults = pm.complete_parameters({} if hyperparameter is None else hyperparameter)
        super().__init__(defaults)
        self._spot = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        For DSpot there is no training step; we only configure the Spot instance.
        """
        params = {} if self._hyperparameter is None else self._hyperparameter
        if ps is None:
            raise RuntimeError("pyspot (pylibspot) is not available in the environment")

        self._spot = ps.Spot(
            q=params.get('q', 1e-3),
            n_init=params.get('n_init', 1000),
            level=params.get('level', 0.99),
            max_excess=params.get('max_excess', 200),
            up=params.get('up', True),
            down=params.get('down', True),
            alert=params.get('alert', True),
            bounded=params.get('bounded', True),
        )
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Run the Spot detector step-by-step and return a binary anomaly array.

        Parameters
        ----------
        x : np.ndarray
            1-D array of values to detect anomalies on.

        Returns
        -------
        events : np.ndarray
            Binary array (1 for anomaly, 0 for normal) of the same length as `x`.
        """
        if self._spot is None:
            # lazy init
            self.fit(np.array([]))

        X = np.asarray(x).reshape(-1)
        events = np.zeros_like(X)
        for i, r in enumerate(X):
            event = self._spot.step(r)
            if event in [1, -1]:
                events[i] = 1
        return events

    def predict(self, x: np.ndarray) -> np.ndarray:
        # same as decision_function for this detector
        return self.decision_function(x).astype(int)
