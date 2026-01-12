from typing import Optional, List, Dict, Any
import numpy as np
import logging
import joblib

from .. import algorithmbase
from .. import utils

try:
	import stumpy as st
except Exception:
	st = None


def nextpow2(x: int) -> int:
	return int(np.ceil(np.log2(x)))


class DAMPPreprocessor:
	"""Preprocessor used by the original DAMP implementation.

	This is a minimal, self-contained version of the original `DAMPPreprocessor`.
	"""

	def __init__(self, m: int, sp_index: int):
		self.m = m
		self.sp_index = sp_index

	def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> np.ndarray:
		X = self._make_2d(X)
		if self._contains_constant_regions(X):
			X = X.copy()
			X += np.arange(len(X)).reshape(-1, 1) / len(X)
		return X

	def _make_2d(self, X: np.ndarray) -> np.ndarray:
		if X.ndim == 1:
			X = X.reshape(-1, 1)
		elif X.ndim < 1 or X.ndim > 2:
			raise ValueError("The preprocessor can handle only array with a dimensionality of 1 or 2")
		return X

	def _contains_constant_regions(self, X: np.ndarray) -> bool:
		one_row = np.ones(X.shape[1]).reshape(1, -1)
		constant_bool = np.diff(np.concatenate([one_row, np.diff(X, axis=0)], axis=0), axis=0) != 0
		for i in range(X.shape[1]):
			constant_indices = np.argwhere(constant_bool[:, i])
			if constant_indices.size == 0:
				constant_length = X.shape[0]
			else:
				if constant_indices.shape[0] < 2:
					constant_length = 0
				else:
					constant_length = np.max(np.diff(constant_indices, axis=0))
			if constant_length >= self.m or np.var(X[:, i]) < 0.2:
				return True
		return False


class _DAMP:
	"""DAMP algorithm ported from TimeEval's implementation.

	Parameters
	----------
	m : int
		Window size (default 50)
	sp_index : int
		Initial training index (default 200)
	x_lag : Optional[int]
		Maximum lag to search (optional)
	golden_batch : Optional[np.ndarray]
		Optional reference batch
	preprocessing : bool
		Whether to apply the DAMP preprocessor
	lookahead : Optional[int]
		Lookahead parameter
	with_prefix_handling : bool
		Enable prefix handling
	"""

	def __init__(self,
				 m: int = 50,
				 sp_index: int = 200,
				 x_lag: Optional[int] = None,
				 golden_batch: Optional[np.ndarray] = None,
				 preprocessing: bool = True,
				 lookahead: Optional[int] = None,
				 with_prefix_handling: bool = True):
		self.m = m
		self.sp_index = sp_index
		self.x_lag = x_lag or 2**nextpow2(8*self.m)
		self.golden_batch = golden_batch
		self.preprocessing = preprocessing
		self.lookahead = int(2**nextpow2(lookahead) if lookahead is not None else 2 ** nextpow2(16 * self.m))
		self.with_prefix_handling = with_prefix_handling

		self._pv: Optional[np.ndarray] = None
		self._amp: Optional[np.ndarray] = None
		self._bsf = 0

	def fit_transform(self, X: np.ndarray, y=None, **fit_params) -> np.ndarray:
		if self.preprocessing:
			preprocessor = DAMPPreprocessor(m=self.m, sp_index=self.sp_index)
			X = preprocessor.fit_transform(X)

		self._pv = np.ones(len(X) - self.m + 1, dtype=int)
		self._amp = np.zeros_like(self._pv, dtype=float)

		if self.with_prefix_handling:
			self._handle_prefix(X)

		for i in range(self.sp_index, len(X) - self.m + 1):
			if not self._pv[i]:
				self._amp[i] = self._amp[i-1]-0.00001
				continue

			self._amp[i] = self._backward_processing(X, i)
			self._forward_processing(X, i)

		return self._amp

	def _backward_processing(self, X: np.ndarray, i) -> float:
		amp_i = np.inf
		prefix = 2**nextpow2(self.m)
		max_lag = min(self.x_lag or i, i)
		reference_ts = self.golden_batch or X[i-max_lag:i]
		first = True
		expansion_num = 0

		while amp_i >= self._bsf:
			if prefix >= max_lag:
				amp_i = min(self._distance(X[i:i+self.m], reference_ts))
				if amp_i > self._bsf:
					self._bsf = amp_i
				break
			else:
				if first:
					first = False
					amp_i = min(self._distance(X[i:i+self.m], reference_ts[-prefix:]))
				else:
					start = i-max_lag+(expansion_num * self.m)
					end = int(i-(max_lag/2)+(expansion_num * self.m))
					amp_i = min(self._distance(X[i:i+self.m], X[start:end]))

				if amp_i < self._bsf:
					break
				else:
					prefix = 2*prefix
					expansion_num += 1

		return amp_i

	def _forward_processing(self, X: np.ndarray, i):
		start = i + self.m
		end = start + self.lookahead
		indices: List[int] = []

		if end < len(X):
			d = self._distance(X[i:i+self.m], X[start:end])
			indices = np.argwhere(d < self._bsf)
			indices += start

		if len(indices) > 0:
			self._pv[indices.flatten()] = 0

	def _distance(self, Q: np.ndarray, T: np.ndarray) -> np.ndarray:
		if st is None:
			raise RuntimeError('stumpy is required for DAMP._distance')
		n_variates = Q.shape[1]
		return np.sum([st.core.mass(Q[:, d], T[:, d]) for d in range(n_variates)], axis=0)

	def _handle_prefix(self, X: np.ndarray):
		for i in range(self.sp_index, min(self.sp_index + (16 * self.m), self._pv.shape[0])):
			if self._pv[i] != 0:
				self._amp[i] = self._amp[i-1]-0.00001
				continue

			if i + self.m > X.shape[0]:
				break

			query = X[i:i+self.m]
			self._amp[i] = min(self._distance(query, X[:i]))
			self._bsf = max(self._amp)

			if self.lookahead > 0:
				start_of_mass = min(i+self.m, X.shape[0])
				end_of_mass = min(start_of_mass+self.lookahead, X.shape[0])

				if (end_of_mass - start_of_mass + 1) > self.m:
					distance_profile = self._distance(query, X[start_of_mass:end_of_mass])
					dp_index_less_than_BSF = np.argwhere(distance_profile < self._bsf)
					ts_index_less_than_BSF = dp_index_less_than_BSF + start_of_mass - 1
					self._pv[ts_index_less_than_BSF.flatten()] = 0


class DAMP(algorithmbase.AlgorithmBase):
	"""Adapter that implements AlgorithmBase using the embedded DAMP.

	Parameters
	----------
	hyperparameter : dict
		Hyperparameters mapped to DAMP's constructor. Recognized keys include
		`m`, `sp_index`, `x_lag`, `preprocessing`, `lookahead`, and
		`with_prefix_handling`.
	"""

	def __init__(self, hyperparameter: Dict[str, Any] = None):
		pm = utils.ParameterManagement(_DAMP.__init__)
		defaults = pm.complete_parameters({} if hyperparameter is None else hyperparameter)
		super().__init__(defaults)
		self._damp: Optional[_DAMP] = None

	def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
		"""
		Fit/create the internal DAMP instance and run `fit_transform` on `x_train`.

		Parameters
		----------
		x_train : np.ndarray
			Time series data, shape (n_samples, n_features) or 1D array.
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
		self._damp = _DAMP(
			m=params.get('m', params.get('anomaly_window_size', 50)),
			sp_index=params.get('sp_index', params.get('n_init_train', 200)),
			x_lag=params.get('x_lag', params.get('max_lag', None)),
			preprocessing=params.get('preprocessing', True),
			lookahead=params.get('lookahead', params.get('lookahead', None)),
			with_prefix_handling=params.get('with_prefix_handling', True)
		)

		# run fit_transform to compute the moving profile
		mp = self._damp.fit_transform(X)
		# store internal model for later use
		self._model = self._damp
		# also store last computed scores
		self._last_scores = mp
		return self

	def decision_function(self, x: np.ndarray) -> np.ndarray:
		"""
		Compute anomaly scores for `x` using the DAMP algorithm.

		Parameters
		----------
		x : np.ndarray
			Input time series, shape (n_samples, n_features) or 1D array.

		Returns
		-------
		scores : np.ndarray
			1-D array with per-timestep anomaly scores.
		"""
		X = np.asarray(x)
		if X.ndim == 1:
			X = X.reshape(-1, 1)

		if self._damp is None:
			# lazy-create a DAMP instance with hyperparameters
			params = {} if self._hyperparameter is None else self._hyperparameter
			self._damp = _DAMP(
				m=params.get('m', params.get('anomaly_window_size', 50)),
				sp_index=params.get('sp_index', params.get('n_init_train', 200)),
				x_lag=params.get('x_lag', params.get('max_lag', None)),
				preprocessing=params.get('preprocessing', True),
				lookahead=params.get('lookahead', params.get('lookahead', None)),
				with_prefix_handling=params.get('with_prefix_handling', True)
			)

		scores = self._damp.fit_transform(X)
		self._last_scores = scores
		return np.asarray(scores)

	def predict(self, x: np.ndarray) -> np.ndarray:
		"""
		Binarize anomaly scores using the 95th percentile as threshold.

		Parameters
		----------
		x : np.ndarray
			Input time series.

		Returns
		-------
		preds : np.ndarray
			Binary array where 1 indicates anomaly.
		"""
		scores = self.decision_function(x)
		thresh = np.percentile(scores, 95)
		return (scores >= thresh).astype(int)
