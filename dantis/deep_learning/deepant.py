from typing import Dict, Any
import numpy as np
import torch

from .. import algorithmbase
from .. import utils

from .wrapper_deepant.predictor import Predictor, retrieve_save_path
from .wrapper_deepant.detector import Detector
from .wrapper_deepant.dataset import TimeSeries


class DeepAnT(algorithmbase.AlgorithmBase):
	"""
	Adapter for DeepAnT implementing the AlgorithmBase interface.

	Parameters
	----------
	hyperparameter : dict
		Hyperparameters accepted include:
		- epochs
		- window_size
		- prediction_window_size
		- learning_rate
		- batch_size
		- split
		- early_stopping_delta
		- early_stopping_patience
		- random_state

	Notes
	-----
	This adapter uses the code inside `wrapper_deepant` (Predictor, Detector,
	TimeSeries). The `Predictor.save`/`load` methods manage model checkpointing.
	"""

	def __init__(self, hyperparameter: Dict[str, Any] = None):
		pm = utils.ParameterManagement(Predictor.__init__)
		defaults = pm.complete_parameters({} if hyperparameter is None else hyperparameter)
		super().__init__(defaults)
		self._predictor: Predictor | None = None
		self._detector: Detector | None = None

	def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
		"""
		Train the DeepAnT predictor on the provided multivariate time series.

		Parameters
		----------
		x_train : np.ndarray
			Array with shape (n_samples, n_features). If `y_train` is provided
			it will be ignored (the model is unsupervised for anomalies).
		y_train : ignored
			Kept for interface compatibility.

		Returns
		-------
		self
		"""
		X = np.asarray(x_train)
		if X.ndim == 1:
			X = X.reshape(-1, 1)

		params = {} if self._hyperparameter is None else self._hyperparameter
		window = int(params.get('window_size', params.get('window', 45)))
		pred_window = int(params.get('prediction_window_size', params.get('pred_window', 1)))
		lr = float(params.get('learning_rate', params.get('lr', 1e-5)))
		batch_size = int(params.get('batch_size', 45))
		epochs = int(params.get('epochs', 50))
		split = float(params.get('split', 0.75))
		early_delta = float(params.get('early_stopping_delta', 0.05))
		early_patience = int(params.get('early_stopping_patience', 10))

		# prepare datasets
		n_samples = X.shape[0]
		train_samples = int(split * n_samples)
		train_dataset = TimeSeries(X[:train_samples], window_length=window, prediction_length=pred_window)
		valid_dataset = TimeSeries(X[train_samples:], window_length=window, prediction_length=pred_window)

		# create predictor and train
		self._predictor = Predictor(window, pred_window, lr=lr, batch_size=batch_size, in_channels=X.shape[1])
		save_path = params.get('modelOutput', 'deepant_model.pt')
		self._predictor.train(train_dataset, valid_dataset, n_epochs=epochs, save_path=save_path,
							  log_freq=10, early_stopping_patience=early_patience, early_stopping_delta=early_delta)

		# after training, load best model into predictor
		try:
			self._predictor.load(save_path)
		except Exception:
			# if load fails, we still keep the in-memory model
			pass

		# detector
		self._detector = Detector()
		self._model = self._predictor
		return self

	def decision_function(self, x: np.ndarray) -> np.ndarray:
		"""
		Compute anomaly scores for input `x` using the predictor and detector.

		Parameters
		----------
		x : np.ndarray
			Array shape (n_samples, n_features).

		Returns
		-------
		scores : np.ndarray
			1-D array with anomaly scores.
		"""
		if self._predictor is None or self._detector is None:
			raise Exception("Model not trained or loaded")

		X = np.asarray(x)
		if X.ndim == 1:
			X = X.reshape(-1, 1)

		window = int(self._hyperparameter.get('window_size', 45))
		pred_window = int(self._hyperparameter.get('prediction_window_size', 1))

		test_dataset = TimeSeries(X, window_length=window, prediction_length=pred_window)
		predictedY = self._predictor.predict(test_dataset)
		anomalies = self._detector.detect(predictedY, test_dataset)
		return np.asarray(anomalies)

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""
		Return binarized anomaly labels from scores (threshold at 95th percentile).
		"""
		scores = self.decision_function(x)
		thresh = np.percentile(scores, 95)
		return (scores >= thresh).astype(int)

	# save_model/load_model default to AlgorithmBase which pickles the predictor/detector


def set_random_state(config) -> None:
	seed = config.get('random_state', config.get('customParameters', {}).get('random_state', 42)) if isinstance(config, dict) else getattr(config, 'random_state', 42)
	import random
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

