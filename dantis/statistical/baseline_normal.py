import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from .. import algorithmbase


class NormalBaseline(algorithmbase.AlgorithmBase):
	"""
	Constant normal baseline anomaly scorer that assigns a constant score of
	0.0 to every sample.

	This baseline does not learn from data. It always returns a vector of
	zeros with the same length as the provided input (or the recorded
	training length when `decision_function(None)` is used).

	Parameters
	----------
	hyperparameter : dict, optional
		Dictionary of model hyperparameters. Kept for API compatibility but not
		used by this baseline.

	Attributes
	----------
	_train_length : int
		Length of the training series recorded during `fit`. Used when
		`decision_function(None)` is called to generate scores for the same
		length.
	"""

	def __init__(self, hyperparameter: dict = None):
		super().__init__(hyperparameter=hyperparameter)
		# store the training length so decision_function(None) can be used
		self._train_length = 0

	def decision_function(self, x):
		"""
		Return a 1-D array of constant zero scores.

		If `x` is None, the stored training length (from `fit`) is used.

		Parameters
		----------
		x : array-like or None
			Input data whose first dimension defines the number of samples.
			If None, the stored training length is used.

		Returns
		-------
		numpy.ndarray
			1-D array of zeros with shape (n_samples,). Returns an empty array
			if `n_samples` is 0.
		"""
		if x is None:
			n = getattr(self, "_train_length", 0)
		else:
			# support 1-D or 2-D arrays: take the first dimension as the
			# temporal length
			try:
				n = int(np.asarray(x).shape[0])
			except Exception:
				n = 0

		if n <= 0:
			return np.array([])

		scores = np.zeros(n).reshape(-1)
		return scores

	def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
		"""
		"Fit" the baseline by recording the training series length.

		No learning is performed. Only the number of samples in `x_train` is
		recorded so that `decision_function(None)` can generate scores for that
		length.

		Parameters
		----------
		x_train : numpy.ndarray
			Training data. Only its length (first dimension) is used.
		y_train : numpy.ndarray, optional
			Unused. Present for API compatibility.

		Returns
		-------
		numpy.ndarray
			Anomaly scores for `x_train`.
		"""
		logging.info("Fitting NormalBaseline (no-op). y_train ignored.")
		self._train_length = 0 if x_train is None else int(np.asarray(x_train).shape[0])
		return self.get_anomaly_score(x_train)

	def get_anomaly_score(self, x: np.ndarray = None):
		"""
		Alias for `decision_function` to comply with the expected API.

		Parameters
		----------
		x : numpy.ndarray or None, optional
			Input data or None to use recorded training length.

		Returns
		-------
		numpy.ndarray
			Anomaly scores produced by `decision_function`.
		"""
		return self.decision_function(x)

	def predict(self, x: np.ndarray = None) -> np.ndarray:
		"""
		Predict anomaly scores for the given input.

		Parameters
		----------
		x : numpy.ndarray or None, optional
			Input data.

		Returns
		-------
		numpy.ndarray
			Anomaly scores.
		"""
		return self.get_anomaly_score(x)

	def save_model(self, path_model):
		"""
		Save model state to the given path.

		For this baseline there is no complex state to serialize beyond what
		the base class may handle.
		"""
		super().save_model(path_model)

	def load_model(self, path_model):
		"""
		Load model state from the given path.
		"""
		super().load_model(path_model)

	def set_hyperparameter(self, hyperparameter):
		"""
		Set hyperparameters for compatibility with the base API.

		This baseline does not require any hyperparameters but stores the
		provided dict for compatibility.
		"""
		self._hyperparameter = hyperparameter or {}