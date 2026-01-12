from typing import Optional, Dict, Any
import pomegranate as pg
from pomegranate.distributions import Categorical
import numpy as np
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.preprocessing import KBinsDiscretizer
from typing import List, Optional, Tuple, Union
from collections import Counter
import networkx as nx
import joblib
from os import PathLike
from .. import algorithmbase
from .. import utils



class _DynamicBN(BaseEstimator, OutlierMixin):
    """
    # Dynamic-Bayesian-Network Anomaly-Detector

    - Discretization
    - (default) Two-Timeslice BN architecture
    """

    def __init__(self, timesteps: int, discretizer_n_bins: int):
        self.features: List[Tuple[int, int]] = []
        self.timesteps = timesteps
        self.discretizer = KBinsDiscretizer(n_bins=discretizer_n_bins, encode='ordinal', strategy='uniform')
        self.bayesian_network: Optional[Union[pg.BayesianNetwork, str]] = None

    def _get_distribution(X: np.ndarray) -> tuple[Categorical, Dict[str, int]]:
        counts = Counter(map(str, X))
        symbols = sorted(counts.keys())  # deterministic order
        symbol_to_idx = {s: i for i, s in enumerate(symbols)}

        total = float(len(X))
        probs = np.array([[counts[s] / total for s in symbols]], dtype=np.float64)

        dist = Categorical(probs=probs)
        return dist, symbol_to_idx

    def _preprocess_data(self, X: np.ndarray, fit: bool) -> np.ndarray:
        X = self.discretizer.fit_transform(X).astype(int) if fit else self.discretizer.transform(X).astype(int)
        rolled_Xs = [X]
        for t in range(1, self.timesteps):
            rolled_Xs.append(np.roll(X, -t, axis=0))
        X = np.concatenate(rolled_Xs, axis=1)[:-(self.timesteps-1)]
        return X

    def _build_contraint_graph(self, n_nodes: int) -> nx.DiGraph:
        """
        Building a graph of possible edges.
        Only edges within a time point and to a future time point are allowed in order to build a DBN.
        """
        constraint_graph = nx.DiGraph()
        constraint_graph.add_nodes_from(list(range(n_nodes)))
        n_nodes_per_time = n_nodes // self.timesteps

        for t0 in range(self.timesteps):
            for n0 in range(n_nodes_per_time):
                node_start = t0 * n_nodes_per_time + n0
                for t1 in range(t0, self.timesteps):
                    for n1 in range(n_nodes_per_time):
                        node_end = t1 * n_nodes_per_time + n1
                        if node_start != node_end:
                            constraint_graph.add_edge(node_start, node_end)

        return constraint_graph

    def fit(self, X: np.ndarray, y=None) -> '_DynamicBN':
        X = self._preprocess_data(X, True)
        constraint_graph = self._build_contraint_graph(X.shape[1])
        self.bayesian_network = pg.BayesianNetwork.from_samples(X, constraint_graph=constraint_graph)
        self.bayesian_network.bake()
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = self._preprocess_data(X, False)
        cut_probs = np.zeros(self.timesteps - 1) + np.nan
        return np.concatenate([-self.bayesian_network.log_probability(X), cut_probs])

    def save(self, path: PathLike):
        self.bayesian_network = self.bayesian_network.to_json()
        joblib.dump(self, path)

    @staticmethod
    def load(path: PathLike) -> '_DynamicBN':
        model: _DynamicBN = joblib.load(path)
        model.bayesian_network = pg.BayesianNetwork.from_json(model.bayesian_network)
        return model

class LaserDBNWrapper(algorithmbase.AlgorithmBase):
    """Wrapper for _DynamicBN.

    Default hyperparameters follow `CustomParameters` in the TimeEval
    `algorithm.py` for this algorithm.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "timesteps": 2,
            "n_bins": 10,
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
        # Try to validate/complete using ParameterManagement; fall back to merged dict
        try:
            pm = utils.ParameterManagement(lambda **kwargs: None)
            pm._default_parameters = defaults.copy()
            validated = pm.check_hyperparameter_type(merged)
            merged = pm.complete_parameters(validated)
        except Exception:
            pass
        self._hyperparameter = merged

    def _create_model(self):
        if _DynamicBN is None:
            raise RuntimeError("_DynamicBN implementation not available (laser_dbn package not found)")
        params = self._hyperparameter
        timesteps = int(params.get("timesteps", 2))
        n_bins = int(params.get("n_bins", 10))
        self._model = _DynamicBN(timesteps=timesteps, discretizer_n_bins=n_bins)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """Fit the DBN and return point-wise anomaly scores for x_train.

        The original `_DynamicBN.predict` returns an array where
        the last `timesteps-1` positions are NaN (cut_probs). We return the full
        score vector as-is.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._create_model()
        X = np.asarray(x_train)
        self._model.fit(X)
        scores = self._model.predict(X)
        # ensure numpy array
        return np.asarray(scores)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        X = np.asarray(x)
        return np.asarray(self._model.predict(X))

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.get_anomaly_score(x)
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        if not (0 < contamination < 0.5):
            contamination = max(0.001, min(0.5, contamination))
        # ignore NaNs when computing threshold
        valid = np.isfinite(scores)
        if not np.any(valid):
            # if everything is NaN, return zeros
            return np.zeros_like(scores, dtype=int)
        threshold = np.percentile(scores[valid], 100.0 * (1.0 - contamination))
        # treat NaN positions as non-anomalous (0)
        labels = np.zeros_like(scores, dtype=int)
        labels[valid] = (scores[valid] >= threshold).astype(int)
        return labels

    def save_model(self, path_model: str):
        super().save_model(path_model)

    def load_model(self, path_model: str):
        super().load_model(path_model)
