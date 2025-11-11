from typing import Optional, Dict, Any
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cluster import KMeans
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


from .. import algorithmbase
from .. import utils


class _KMeansAD(BaseEstimator, OutlierMixin):
    def __init__(self, k: int, window_size: int, stride: int, n_jobs: int):
        self.k = k
        self.window_size = window_size
        self.stride = stride
        self.model = KMeans(n_clusters=k, n_jobs=n_jobs)
        self.padding_length = 0

    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        flat_shape = (X.shape[0] - (self.window_size - 1), -1)  # in case we have a multivariate TS
        slides = sliding_window_view(X, window_shape=self.window_size, axis=0).reshape(flat_shape)[::self.stride, :]
        self.padding_length = X.shape[0] - (slides.shape[0] * self.stride + self.window_size - self.stride)
        print(f"Required padding_length={self.padding_length}")
        return slides

    def _custom_reverse_windowing(self, scores: np.ndarray) -> np.ndarray:
        print("Reversing window-based scores to point-based scores:")
        print(f"Before reverse-windowing: scores.shape={scores.shape}")
        # compute begin and end indices of windows
        begins = np.array([i * self.stride for i in range(scores.shape[0])])
        ends = begins + self.window_size

        # prepare target array
        unwindowed_length = self.stride * (scores.shape[0] - 1) + self.window_size + self.padding_length
        mapped = np.full(unwindowed_length, fill_value=np.nan)

        # only iterate over window intersections
        indices = np.unique(np.r_[begins, ends])
        for i, j in zip(indices[:-1], indices[1:]):
            window_indices = np.flatnonzero((begins <= i) & (j-1 < ends))
            # print(i, j, window_indices)
            mapped[i:j] = np.nanmean(scores[window_indices])

        # replace untouched indices with 0 (especially for the padding at the end)
        np.nan_to_num(mapped, copy=False)
        print(f"After reverse-windowing: scores.shape={mapped.shape}")
        return mapped

    def fit(self, X: np.ndarray, y=None, preprocess=True) -> 'KMeansAD':
        if preprocess:
            X = self._preprocess_data(X)
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray, preprocess=True) -> np.ndarray:
        if preprocess:
            X = self._preprocess_data(X)
        clusters = self.model.predict(X)
        diffs = np.linalg.norm(X - self.model.cluster_centers_[clusters], axis=1)
        return self._custom_reverse_windowing(diffs)

    def fit_predict(self, X, y=None) -> np.ndarray:
        X = self._preprocess_data(X)
        self.fit(X, y, preprocess=False)
        return self.predict(X, preprocess=False)


class KMeansAD(algorithmbase.AlgorithmBase):
    """Wrapper around the windowed KMeans anomaly detector.

    Hyperparameters (defaults follow TimeEval's `CustomParameters`):
    - n_clusters
    - anomaly_window_size
    - stride
    - n_jobs
    - random_state
    - contamination
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "n_clusters": 20,
            "anomaly_window_size": 20,
            "stride": 1,
            "n_jobs": 1,
            "random_state": 42,
            "contamination": 0.1,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._model = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults
            return
        merged = defaults.copy()
        merged.update(hyperparameter)
        # validate/complete using ParameterManagement
        try:
            pm = utils.ParameterManagement(lambda: None)
            pm._default_parameters = defaults.copy()
            validated = pm.check_hyperparameter_type(merged)
            merged = pm.complete_parameters(validated)
        except Exception:
            pass
        self._hyperparameter = merged

    def _create_model(self):
        if _KMeansAD is None:
            raise RuntimeError("KMeansAD implementation not available (kmeans package not found)")
        params = self._hyperparameter
        k = int(params.get("n_clusters", 20))
        window_size = int(params.get("anomaly_window_size", 20))
        stride = int(params.get("stride", 1))
        n_jobs = int(params.get("n_jobs", 1))
        self._model = _KMeansAD(k=k, window_size=window_size, stride=stride, n_jobs=n_jobs)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """Fit the windowed KMeans model and return anomaly scores for x_train."""
        self.set_hyperparameter(self._hyperparameter)
        self._create_model()
        X = np.asarray(x_train)
        # fit and then predict to produce point-wise scores
        self._model.fit(X)
        scores = self._model.predict(X)
        return scores

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        X = np.asarray(x)
        return self._model.predict(X)

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.get_anomaly_score(x)
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        if not (0 < contamination < 0.5):
            contamination = max(0.001, min(0.5, contamination))
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        return (scores >= threshold).astype(int)

    def save_model(self, path_model: str):
        super().save_model(path_model)

    def load_model(self, path_model: str):
        super().load_model(path_model)
        # If the saved object was the underlying detector, place it in _model
        if isinstance(self._model, _KMeansAD):
            # nothing to do
            pass
