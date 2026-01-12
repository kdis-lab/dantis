from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

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


logger = logging.getLogger(__name__)


class DSpot(algorithmbase.AlgorithmBase):
    """DSpot wrapper (pyspot) adapted to DANTIS conventions.

    - Uses `utils.ParameterManagement` to coerce/complete hyperparameters.
    - Exposes `get_default_hyperparameters` so detectors are instantiable without args.
    - `fit` creates/configures the `ps.Spot` instance; optionally warms it with `x_train`.
    - `decision_function` returns float anomaly scores (0.0/1.0); `predict` thresholds using `contamination`.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            'q': 1e-3,
            'n_init': 1000,
            'level': 0.99,
            'max_excess': 200,
            'up': True,
            'down': True,
            'alert': True,
            'bounded': True,
            'random_state': 42,
            'warmup': True,  # whether to step through x_train in fit
            'contamination': 0.01,
        }

    def __init__(self, hyperparameter: Dict[str, Any] = None):
        # Create a template function for ParameterManagement
        def _tpl(q=1e-3, n_init=1000, level=0.99, max_excess=200, up=True, down=True, alert=True, bounded=True, random_state=42, warmup=True, contamination=0.01):
            return None

        pm = utils.ParameterManagement(_tpl)
        coerced = pm.check_hyperparameter_type(hyperparameter or {})
        merged = pm.complete_parameters(coerced)
        super().__init__(merged)

        self._spot: Optional[object] = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        if hyperparameter is None:
            self._hyperparameter = self.get_default_hyperparameters()
            return
        def _tpl(q=1e-3, n_init=1000, level=0.99, max_excess=200, up=True, down=True, alert=True, bounded=True, random_state=42, warmup=True, contamination=0.01):
            return None
        pm = utils.ParameterManagement(_tpl)
        coerced = pm.check_hyperparameter_type(hyperparameter)
        merged = pm.complete_parameters(coerced)
        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v
        self._hyperparameter = merged

    def _init_spot(self):
        if ps is None:
            raise ImportError("pyspot is required for DSpot but is not available")
        params = self._hyperparameter or self.get_default_hyperparameters()
        if self._spot is None:
            self._spot = ps.Spot(
                q=params.get('q', 1e-3),
                n_init=int(params.get('n_init', 1000)),
                level=float(params.get('level', 0.99)),
                max_excess=int(params.get('max_excess', 200)),
                up=bool(params.get('up', True)),
                down=bool(params.get('down', True)),
                alert=bool(params.get('alert', True)),
                bounded=bool(params.get('bounded', True)),
            )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """Configure Spot and optionally warm it with `x_train`.

        Returns
        -------
        self
        """
        if ps is None:
            raise RuntimeError("pyspot (pylibspot) is not available in the environment")

        self._init_spot()
        params = self._hyperparameter or self.get_default_hyperparameters()
        warmup = bool(params.get('warmup', True))

        if warmup and x_train is not None and len(x_train) > 0:
            X = np.asarray(x_train).reshape(-1)
            for v in X:
                # ensure numeric
                try:
                    self._spot.step(float(v))
                except Exception:
                    logger.warning('pyspot.step failed for value %r', v)

        # keep a reference for save/load
        self._model = self._spot
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Run the detector and return float scores in [0,1]."""
        if self._spot is None:
            # lazy init
            self.fit(np.array([]))

        X = np.asarray(x).reshape(-1)
        scores = np.zeros(X.shape[0], dtype=float)
        for i, v in enumerate(X):
            try:
                event = self._spot.step(float(v))
            except Exception:
                event = 0
            if event in (1, -1):
                scores[i] = 1.0
            else:
                scores[i] = 0.0
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        contamination = float(self._hyperparameter.get('contamination', 0.01))
        contamination = max(0.001, min(0.5, contamination))
        if scores.size == 0:
            return np.zeros(0, dtype=int)
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        return (scores >= threshold).astype(int)

    def save_model(self, path: str):
        try:
            import joblib
        except Exception:
            raise ImportError('save_model requires joblib')

        payload = {'hyperparameter': self._hyperparameter}
        # try to persist internal spot if picklable
        try:
            payload['spot'] = self._spot
        except Exception:
            payload['spot'] = None
        joblib.dump(payload, path)

    def load_model(self, path: str):
        try:
            import joblib
        except Exception:
            raise ImportError('load_model requires joblib')
        payload = joblib.load(path)
        self._hyperparameter = payload.get('hyperparameter', self.get_default_hyperparameters())
        spot_obj = payload.get('spot')
        if spot_obj is not None:
            self._spot = spot_obj
        return self
