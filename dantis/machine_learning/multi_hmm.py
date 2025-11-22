from typing import Optional, Dict, Any, Tuple, Union

import numpy as np
import pandas as pd

from .. import algorithmbase
from .. import utils

import numpy.typing as npt

import pomegranate as pg

from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from fcmeans import FCM


"""
Inspired by https://github.com/Fuminides/Fancy_aggregations/blob/master/Fancy_aggregations/integrals.py
"""
class BaseFuzzyIntegrator(BaseEstimator, TransformerMixin):
    def __init__(self, measure: Optional[npt.ArrayLike] = None, axis: int = 0, p: int = 2):
        self.measure = measure
        self.axis = axis
        self.p = p

    def _generate_cardinality(self, n: int) -> npt.ArrayLike:
        card = np.arange(n, 0, -1)
        return (card / n) ** self.p

    def fit(self, X: npt.ArrayLike, y=None) -> 'BaseFuzzyIntegrator':
        if self.measure is None:
            self.measure = self._generate_cardinality(X.shape[self.axis])
            self.measure = np.expand_dims(self.measure, axis=1 - self.axis)
        return self

    def transform(self, X: npt.ArrayLike, y=None) -> npt.ArrayLike:
        assert self.measure is not None, f"Please fit {type(self).__name__} Transformer before transforming!"

        X_sorted = np.sort(X, axis=self.axis)
        return X_sorted


class Sugeno(BaseFuzzyIntegrator):
    def transform(self, X: npt.ArrayLike, y=None) -> npt.ArrayLike:
        X_sorted = super().transform(X, y)

        return np.amax(
            np.minimum(
                np.take(X_sorted, np.arange(0, X_sorted.shape[self.axis]), self.axis),
                self.measure
            ),
            axis=self.axis,
            keepdims=True
        )


class Choquet(BaseFuzzyIntegrator):
    def transform(self, X: npt.ArrayLike, y=None) -> npt.ArrayLike:
        X_sorted = super().transform(X, y)

        X_differenced = np.concatenate([
            np.take(X_sorted, [0], self.axis),
            np.diff(X_sorted, axis=self.axis)
        ], axis=self.axis)
        X_agg = np.dot(X_differenced.transpose((1 - self.axis, self.axis)), self.measure.reshape(-1))

        return X_agg.reshape(-1, 1)


class WrappedFCM(BaseEstimator, TransformerMixin):
    def __init__(self, n_bins: int):
        self.fcm = FCM(n_clusters=n_bins)

    @property
    def n_bins(self) -> int:
        return self.fcm.n_clusters

    def fit(self, X: npt.ArrayLike, y=None) -> 'WrappedFCM':
        self.fcm.fit(X)
        return self

    def transform(self, X: npt.ArrayLike, y=None) -> npt.ArrayLike:
        return self.fcm.predict(X).reshape(-1, 1)

    def fit_transform(self, X, y=None, **fit_params) -> npt.ArrayLike:
        self.fit(X)
        return self.transform(X)


class _MultiHMM(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.hmm: Optional[Union[pg.HiddenMarkovModel, str]] = None

    def fit(self, X: npt.ArrayLike, y: npt.ArrayLike) -> '_MultiHMM':
        y = ['None-start'] + y.tolist() + ['None-end']
        self.hmm = pg.HiddenMarkovModel.from_samples(pg.NormalDistribution, n_components=2, X=X)
        self.hmm.bake()

        self.hmm.fit(X, labels=y, algorithm="viterbi")
        return self

    def predict(self, X: npt.ArrayLike) -> npt.ArrayLike:
        log_proba = self.hmm.predict_log_proba(X)
        return log_proba[:, 1]

    def __getstate__(self):
        self.hmm = self.hmm.to_json()
        return super().__getstate__()

    def __setstate__(self, state):
        super().__setstate__(state)
        self.hmm = pg.HiddenMarkovModel.from_json(self.hmm)


class MultiHMMADBuilder:
    def __init__(self, n_bins: int, discretizer: str, n_features: int):
        self.n_bins = n_bins
        self.discretizer = discretizer
        self.n_features = n_features

    def _build_discretizer(self):
        if self.n_features == 1:
            return KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform')

        if self.discretizer == "sugeno":
            return Pipeline([
                ("Sugeno", Sugeno(axis=1)),
                ("Discretization", KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform'))
            ])
        elif self.discretizer == "choquet":
            return Pipeline([
                ("Choquet", Choquet(axis=1)),
                ("Discretization", KBinsDiscretizer(n_bins=self.n_bins, encode='ordinal', strategy='uniform'))
            ])
        else:  # if self.discretizer == "fcm"
            return WrappedFCM(n_bins=self.n_bins)

    def build(self) -> Pipeline:
        algorithm = Pipeline([
            ("StandardScaler", StandardScaler()),
            ("Discretizer", self._build_discretizer()),
            ("MultiHMM", _MultiHMM())
        ])

        return algorithm


class MultiHMM(algorithmbase.AlgorithmBase):
    """Adapter around TimeEval's MultiHMM pipeline.

    Hyperparameters (defaults from TimeEval):
      - n_bins: int
      - discretizer: str ("fcm", "sugeno", "choquet")
      - random_state: int
      - contamination: float
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "n_bins": 10,
            "discretizer": "fcm",
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
        try:
            pm = utils.ParameterManagement(lambda **kwargs: None)
            pm._default_parameters = defaults.copy()
            validated = pm.check_hyperparameter_type(merged)
            merged = pm.complete_parameters(validated)
        except Exception:
            pass
        self._hyperparameter = merged

    def _extract_X_y(self, x: any, y: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Extract features and labels from DataFrame or accept ndarray + y.

        If `x` is a pandas DataFrame and `y` is None this follows the
        TimeEval convention: features = df.iloc[:, 1:-1], labels = df.iloc[:, -1].
        """
        if isinstance(x, pd.DataFrame):
            df = x
            X = df.iloc[:, 1:-1].values.astype(float)
            if y is None:
                y = df.iloc[:, -1].values.astype(str)
            return X, y

        X = np.asarray(x)
        if y is None:
            return X, None
        return X, np.asarray(y)

    def _create_model(self, n_features: int):
        if MultiHMMADBuilder is None:
            raise RuntimeError("multi_hmm package not available in PYTHONPATH")
        params = self._hyperparameter
        builder = MultiHMMADBuilder(n_bins=int(params.get("n_bins", 10)),
                                    discretizer=str(params.get("discretizer", "fcm")),
                                    n_features=int(n_features))
        self._model = builder.build()

    def fit(self, x_train: any, y_train: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the MultiHMM pipeline. Requires labels either via `y_train`
        or embedded in a DataFrame input.
        Returns per-sample anomaly scores on the training data.
        """
        self.set_hyperparameter(self._hyperparameter)
        X, y = self._extract_X_y(x_train, y_train)
        if y is None:
            raise ValueError("MultiHMM.fit requires labels. Pass `y_train` or a DataFrame with labels in the last column.")

        self._create_model(n_features=X.shape[1])
        # The pipeline's fit expects X, y
        self._model.fit(X, y)
        scores = self._model.predict(X)
        return np.asarray(scores)

    def decision_function(self, x: any) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        X, _ = self._extract_X_y(x, None)
        return np.asarray(self._model.predict(X))

    def predict(self, x: any) -> np.ndarray:
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
