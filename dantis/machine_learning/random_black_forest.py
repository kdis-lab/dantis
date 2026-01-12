from typing import Optional, Dict, Any

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import paired_distances

from numpy.lib.stride_tricks import sliding_window_view

from .. import algorithmbase
from .. import utils


class SlidingWindowProcessor(BaseEstimator, TransformerMixin):
    """Utility to create sliding windows and inverse-transform predictions.

    """

    def __init__(self, window_size: int, standardize: bool = False):
        self.window_size = window_size
        self.scaler = StandardScaler() if standardize else None

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **fit_params):
        if self.scaler is not None:
            self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        if self.scaler is not None:
            X = self.scaler.transform(X)
        new_X = sliding_window_view(X, window_shape=self.window_size, axis=0)[:-1]
        new_X = new_X.reshape(new_X.shape[0], -1)
        new_y = np.roll(X, -self.window_size, axis=0)[:-self.window_size]
        return new_X, new_y

    def transform_y(self, X: np.ndarray) -> np.ndarray:
        if self.scaler is not None:
            X = self.scaler.transform(X)
        return np.roll(X, -self.window_size, axis=0)[:-self.window_size]

    def inverse_transform_y(self, y: np.ndarray) -> np.ndarray:
        result = np.full((self.window_size + y.shape[0], y.shape[1]), np.nan)
        result[-len(y) :, :] = y
        if self.scaler is not None:
            result = self.scaler.inverse_transform(result)
        return result


class RandomBlackForestModel(BaseEstimator, RegressorMixin):
    """Internal Random Black Forest model (self-contained).
    """

    def __init__(
        self,
        train_window_size: int = 50,
        n_estimators: int = 2,
        max_features_per_estimator: float = 0.5,
        n_trees: int = 100,
        max_features_method: str = "auto",
        bootstrap: bool = True,
        max_samples: Optional[float] = None,
        standardize: bool = False,
        random_state: int = 42,
        verbose: int = 0,
        n_jobs: int = 1,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ):
        self.preprocessor = SlidingWindowProcessor(train_window_size, standardize)
        self.clf = BaggingRegressor(
            base_estimator=RandomForestRegressor(
                n_estimators=n_trees,
                max_features=max_features_method,
                bootstrap=bootstrap,
                max_samples=max_samples,
                random_state=random_state,
                verbose=verbose,
                n_jobs=n_jobs,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
            ),
            n_estimators=n_estimators,
            max_features=max_features_per_estimator,
            bootstrap_features=False,
            max_samples=1.0,
            n_jobs=n_jobs,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RandomBlackForestModel":
        if y is not None:
            warnings.warn("y is calculated from X and will be ignored if provided")
        Xw, yw = self.preprocessor.fit_transform(X)
        self.clf.fit(Xw, yw)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        Xw, _ = self.preprocessor.transform(X)
        y_hat = self._predict_internal(Xw)
        return self.preprocessor.inverse_transform_y(y_hat)

    def detect(self, X: np.ndarray) -> np.ndarray:
        result_target_shape = X.shape[0]
        Xw, yw = self.preprocessor.transform(X)
        y_hat = self._predict_internal(Xw)
        scores = paired_distances(yw, y_hat.reshape(yw.shape))
        results = np.full(result_target_shape, np.nan)
        results[-len(scores) :] = scores
        return results

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)

    def save(self, path: Path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "RandomBlackForestModel":
        return joblib.load(path)


class RandomBlackForest(algorithmbase.AlgorithmBase):
    """Adapter adapting the internal RandomBlackForestModel to AlgorithmBase."""
    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "train_window_size": 50,
            "n_estimators": 2,
            "max_features_per_estimator": 0.5,
            "n_trees": 100,
            "max_features_method": "auto",
            "bootstrap": True,
            "max_samples": None,
            "standardize": False,
            "random_state": 42,
            "verbose": 0,
            "n_jobs": 1,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "contamination": 0.1,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._model: Optional[RandomBlackForestModel] = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults
            return
        merged = defaults.copy()
        merged.update(hyperparameter)
        try:
            pm = utils.ParameterManagement(lambda **kwargs: None)
            pm._default_parameters = defaults.copy()
            validated = pm.check_hyperparameter_type(merged)
            merged = pm.complete_parameters(validated)
        except Exception:
            pass
        self._hyperparameter = merged

    def _extract_matrix(self, x: Any) -> np.ndarray:
        if isinstance(x, pd.DataFrame):
            df = x
            if df.shape[1] >= 3:
                return df.iloc[:, 1:-1].values.astype(float)
            return df.iloc[:, :-1].values.astype(float)
        arr = np.asarray(x)
        if arr.ndim == 1:
            return arr.reshape(-1, 1).astype(float)
        return arr.astype(float)

    def _create_model(self):
        params = self._hyperparameter
        model_kwargs = {k: params.get(k) for k in params.keys() if k != "contamination"}
        self._model = RandomBlackForestModel(**model_kwargs)

    def fit(self, x_train: Any, y_train: Optional[np.ndarray] = None) -> np.ndarray:
        self.set_hyperparameter(self._hyperparameter)
        self._create_model()
        X = self._extract_matrix(x_train)
        self._model.fit(X)
        scores = self._model.detect(X)
        return np.asarray(scores)

    def decision_function(self, x: Any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        X = self._extract_matrix(x)
        return np.asarray(self._model.detect(X))

    def predict(self, x: Any) -> np.ndarray:
        scores = self.get_anomaly_score(x)
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        if not (0 < contamination < 0.5):
            contamination = max(0.001, min(0.5, contamination))
        valid = np.isfinite(scores)
        if not np.any(valid):
            return np.zeros_like(scores, dtype=int)
        threshold = np.percentile(scores[valid], 100.0 * (1.0 - contamination))
        labels = np.zeros_like(scores, dtype=int)
        labels[valid] = (scores[valid] >= threshold).astype(int)
        return labels

    def save_model(self, path_model: str):
        if self._model is not None:
            self._model.save(Path(path_model))
        else:
            super().save_model(path_model)

    def load_model(self, path_model: str):
        # load underlying model if file contains it
        try:
            loaded = RandomBlackForestModel.load(Path(path_model))
            self._model = loaded
        except Exception:
            # fallback to AlgorithmBase behavior
            super().load_model(path_model)

