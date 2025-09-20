from sklearn.preprocessing import MinMaxScaler
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from .. import algorithmbase


class IncreasingBaseline(algorithmbase.AlgorithmBase):
	"""
	Increasing baseline anomaly scorer that assigns a monotonically increasing
	score (normalized to [0, 1]) according to the temporal index of each
	sample.

	This baseline does not learn from data. It simply produces a deterministic
	increasing sequence of scores based on the position (0 .. n-1) of each
	sample and scales it to the interval [0, 1] using a `MinMaxScaler`.

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
		Return a 1-D array of normalized increasing scores according to the
		temporal index of each sample. If `x` is None, the stored training
		length (from `fit`) is used.

		Parameters
		----------
		x : array-like or None
			Input data whose first dimension defines the number of samples.
			If None, the stored training length is used.

		Returns
		-------
		numpy.ndarray
			1-D array of anomaly scores with shape (n_samples,). Returns an
			empty array if `n_samples` is 0.
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

		indices = np.arange(n).reshape(-1, 1)
		scores = MinMaxScaler().fit_transform(indices).reshape(-1)
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
		logging.info("Fitting IncreasingBaseline (no-op). y_train ignored.")
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