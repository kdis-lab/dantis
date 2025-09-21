from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import Dict, Any
import numpy as np
import torch

from .wrapper_deepnap.model import DeepNAP as _DeepNAP

from .. import algorithmbase
from .. import utils


@dataclass
class _CustomParameters:
	anomaly_window_size: int = 15
	partial_sequence_length: int = 3
	lstm_layers: int = 2
	rnn_hidden_size: int = 200
	dropout: float = 0.5
	linear_hidden_size: int = 100
	batch_size: int = 32
	epochs: int = 1
	learning_rate: float = 0.001
	split: float = 0.8
	early_stopping_delta: float = 0.05
	early_stopping_patience: int = 10
	validation_batch_size: int = 256
	random_state: int = 42


class DeepNAPAdapter(algorithmbase.AlgorithmBase):
	"""
	Adapter for the DeepNAP model that conforms to `AlgorithmBase`.

	Parameters
	----------
	hyperparameter : dict
		Hyperparameters passed to the internal DeepNAP model. If omitted,
		defaults from `_CustomParameters` are used.
	"""

	def __init__(self, hyperparameter: Dict[str, Any] = None):
		pm = utils.ParameterManagement(_DeepNAP.__init__)
		defaults = pm.complete_parameters({} if hyperparameter is None else hyperparameter)
		super().__init__(defaults)
		self._model: _DeepNAP | None = None

	def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
		"""
		Train DeepNAP on `x_train`.

		Parameters
		----------
		x_train : np.ndarray
			Time series array with shape (n_samples, n_features).
		y_train : ignored
			Included for API compatibility.

		Returns
		-------
		self
		"""
		if torch is None:
			raise RuntimeError("PyTorch not available in environment")

		X = np.asarray(x_train)
		if X.ndim == 1:
			X = X.reshape(-1, 1)

		input_size = X.shape[1]

		params = {} if self._hyperparameter is None else self._hyperparameter

		# build args object expected by wrapper DeepNAP.fit/save
		custom_params = _CustomParameters(**{k: params[k] for k in params.keys() & _CustomParameters.__annotations__.keys()})
		args = SimpleNamespace()
		args.customParameters = custom_params
		args.modelOutput = params.get('modelOutput', 'deepnap_model.pt')

		# instantiate model
		model = _DeepNAP(input_size=input_size, **asdict(custom_params))

		# fit and save via wrapper API (wrapper handles early stopping and callbacks)
		model.fit(X, args)
		# wrapper saves model via callback to args.modelOutput; if not, save explicitly
		try:
			model.save(args)
		except Exception:
			pass

		self._model = model
		self._model_path = args.modelOutput
		return self

	def decision_function(self, x: np.ndarray) -> np.ndarray:
		"""
		Compute anomaly scores using the trained DeepNAP model.

		Parameters
		----------
		x : np.ndarray
			Array shape (n_samples, n_features).

		Returns
		-------
		scores : np.ndarray
			Reconstruction-based anomaly scores (higher means more anomalous).
		"""
		if self._model is None:
			raise Exception("Model not trained or loaded")

		X = np.asarray(x)
		if X.ndim == 1:
			X = X.reshape(-1, 1)

		scores = self._model.anomaly_detection(X)
		return np.asarray(scores)

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""
		Binarize anomaly scores using 95th percentile threshold.
		"""
		scores = self.decision_function(x)
		thresh = np.percentile(scores, 95)
		return (scores >= thresh).astype(int)

	def save_model(self, path_model: str):
		"""Save the underlying DeepNAP model using its `save` method if available."""
		if self._model is None:
			raise Exception("No model to save")
		args = SimpleNamespace()
		args.customParameters = _CustomParameters(**(self._hyperparameter or {}))
		args.modelOutput = path_model
		self._model.save(args)

	def load_model(self, path_model: str):
		"""Load a DeepNAP model from disk using the wrapper's `load` staticmethod."""
		args = SimpleNamespace()
		args.modelInput = path_model
		model = _DeepNAP.load(args)
		self._model = model


def set_random_state(config) -> None:
	"""Set seeds for reproducibility.

	`config` can be either an object with `customParameters.random_state` or a
	dict-like containing `random_state`.
	"""
	if isinstance(config, dict):
		seed = config.get('random_state', config.get('customParameters', {}).get('random_state', 42))
	else:
		seed = getattr(config, 'customParameters', None)
		if seed is not None:
			seed = getattr(config.customParameters, 'random_state', 42)
		else:
			seed = getattr(config, 'random_state', 42)

	import random
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

