"""
EncDecAD wrapper for DANTIS.

Adapts the provided PyTorch `EncDecAD` implementation (in
`wrapper_encdec_ad.model`) to the `AlgorithmBase` interface used across
DANTIS. Provides hyperparameter defaults, parameter validation via
`utils.ParameterManagement`, `fit`, `decision_function`, `predict`, and
`save_model`/`load_model` helpers that combine `joblib` and `torch`.
"""
from typing import Optional, Dict, Any
import os
import tempfile
import logging

import numpy as np

try:
    import torch
except Exception:
    torch = None

try:
    import joblib
except Exception:
    joblib = None

from .. import algorithmbase
from .. import utils

from .wrapper_encdec_ad.model import EncDecAD

logger = logging.getLogger('encdec_ad_dantis')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)


class EncDecADDetector(algorithmbase.AlgorithmBase):
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            'input_size': 1,
            'latent_size': 32,
            'lstm_layers': 2,
            'split': 0.8,
            'anomaly_window_size': 50,
            'batch_size': 32,
            'validation_batch_size': 32,
            'test_batch_size': 32,
            'epochs': 10,
            'early_stopping_delta': 1e-4,
            'early_stopping_patience': 3,
            'learning_rate': 1e-3,
            'device': 'cpu',
            'verbose': False,
            'contamination': 0.1,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._model: Optional[EncDecAD] = None
        self._last_scores: Optional[np.ndarray] = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults.copy()
            return

        def _hp_template(input_size=1, latent_size=32, lstm_layers=2, split=0.8, anomaly_window_size=50,
                         batch_size=32, validation_batch_size=32, test_batch_size=32, epochs=10,
                         early_stopping_delta=1e-4, early_stopping_patience=3, learning_rate=1e-3,
                         device='cpu', verbose=False, contamination=0.1):
            return None

        pm = utils.ParameterManagement(_hp_template)
        try:
            coerced = pm.check_hyperparameter_type(hyperparameter)
        except Exception as e:
            raise ValueError(f"EncDecAD Hyperparameter Error: {e}")
        merged = pm.complete_parameters(coerced)
        for k, v in hyperparameter.items():
            if k not in merged:
                merged[k] = v
        self._hyperparameter = merged

    def _ensure_2d(self, series: np.ndarray) -> np.ndarray:
        arr = np.asarray(series)
        if arr.ndim == 1:
            return arr.reshape(-1, 1)
        return arr

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        ts = self._ensure_2d(x_train)
        hp = self._hyperparameter or self.get_default_hyperparameters()

        # instantiate model
        model_kwargs = {
            'input_size': int(hp.get('input_size', ts.shape[1])),
            'latent_size': int(hp.get('latent_size')),
            'lstm_layers': int(hp.get('lstm_layers')),
            'split': float(hp.get('split')),
            'anomaly_window_size': int(hp.get('anomaly_window_size')),
            'batch_size': int(hp.get('batch_size')),
            'validation_batch_size': int(hp.get('validation_batch_size')),
            'test_batch_size': int(hp.get('test_batch_size')),
            'epochs': int(hp.get('epochs')),
            'early_stopping_delta': float(hp.get('early_stopping_delta')),
            'early_stopping_patience': int(hp.get('early_stopping_patience')),
            'learning_rate': float(hp.get('learning_rate')),
        }

        # ensure input_size matches series features when not explicitly provided
        if model_kwargs['input_size'] is None or model_kwargs['input_size'] == 1:
            model_kwargs['input_size'] = ts.shape[1]

        if torch is None:
            raise ImportError('PyTorch is required to run EncDecAD')

        model = EncDecAD(**model_kwargs)

        # prepare a safe path for the underlying PyTorch checkpoint
        tmpdir = tempfile.mkdtemp(prefix='encdec_ad_')
        model_path = os.path.join(tmpdir, 'encdec_ad.pth')

        model.fit(ts, model_path=model_path, verbose=bool(hp.get('verbose', False)))

        # compute anomaly scores for the full series and cache
        try:
            scores = model.anomaly_detection(ts)
        except Exception:
            scores = np.array([])

        self._model = model
        self._last_scores = scores
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        ts = self._ensure_2d(x)
        if self._model is None:
            # train on the fly using current hyperparameters
            self.fit(ts)

        if self._model is None:
            raise RuntimeError('Model not available for decision_function')

        scores = self._model.anomaly_detection(ts)
        self._last_scores = scores
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.decision_function(x)
        contamination = float(self._hyperparameter.get('contamination', 0.1)) if self._hyperparameter else 0.1
        contamination = max(0.001, min(0.5, contamination))
        if scores.size == 0:
            return np.zeros(0, dtype=int)
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        labels = (scores > threshold).astype(int)
        return labels

    def save_model(self, path: str):
        if joblib is None:
            raise ImportError('save_model requires joblib')
        # Save torch state dict to sidecar file and dump hyperparams + state path via joblib
        if torch is None:
            raise ImportError('PyTorch required to save model state')

        base, _ = os.path.splitext(path)
        state_path = base + '.pth'
        payload = {
            'hyperparameter': self._hyperparameter,
            'state_path': state_path,
        }

        if self._model is not None:
            try:
                torch.save({'state_dict': self._model.state_dict()}, state_path)
            except Exception:
                logger.warning('Could not save torch model state')

        joblib.dump(payload, path)

    def load_model(self, path: str):
        if joblib is None:
            raise ImportError('load_model requires joblib')
        if torch is None:
            raise ImportError('PyTorch required to load model state')

        payload = joblib.load(path)
        self._hyperparameter = payload.get('hyperparameter', self.get_default_hyperparameters())
        state_path = payload.get('state_path')
        if state_path and os.path.exists(state_path):
            # instantiate model with current hyperparameters
            hp = self._hyperparameter
            model_kwargs = {
                'input_size': int(hp.get('input_size', 1)),
                'latent_size': int(hp.get('latent_size')),
                'lstm_layers': int(hp.get('lstm_layers')),
                'split': float(hp.get('split')),
                'anomaly_window_size': int(hp.get('anomaly_window_size')),
                'batch_size': int(hp.get('batch_size')),
                'validation_batch_size': int(hp.get('validation_batch_size')),
                'test_batch_size': int(hp.get('test_batch_size')),
                'epochs': int(hp.get('epochs')),
                'early_stopping_delta': float(hp.get('early_stopping_delta')),
                'early_stopping_patience': int(hp.get('early_stopping_patience')),
                'learning_rate': float(hp.get('learning_rate')),
            }
            model = EncDecAD(**model_kwargs)
            ckpt = torch.load(state_path)
            if 'state_dict' in ckpt:
                model.load_state_dict(ckpt['state_dict'])
            else:
                try:
                    model.load_state_dict(ckpt)
                except Exception:
                    logger.warning('Unexpected checkpoint format when loading EncDecAD')
            self._model = model

        return self
