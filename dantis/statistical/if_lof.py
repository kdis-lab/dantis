import math
import heapq
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from .. import algorithmbase
from .. import utils


def outlier_coefficient_for_attribute(attr_index: int, data: np.ndarray) -> float:
    attr = data[:, attr_index]
    mean = np.mean(attr)
    esd = np.std(attr)
    if mean == 0:
        # avoid division by zero; if mean is zero, use a large coefficient
        return float("inf") if esd > 0 else 0.0
    return float(np.abs(esd / mean))


def compute_iforest_scores(X: np.ndarray, n_trees: int, max_samples: Optional[float], random_state: int) -> np.ndarray:
    max_samples_param = max_samples if max_samples is not None else "auto"
    forest = IsolationForest(n_estimators=int(n_trees), max_samples=max_samples_param, random_state=int(random_state))
    forest.fit(X)
    scores = forest.decision_function(X)
    # original implementation negated scores so that larger means more anomalous
    return -scores


def prune_data(data: np.ndarray, anomaly_scores: np.ndarray, alpha: float, m: Optional[int]) -> Tuple[List[np.ndarray], List[int]]:
    n_features = data.shape[1]
    if m is None:
        m = n_features

    # compute outlier coefficients per attribute
    outlier_coefficients = [outlier_coefficient_for_attribute(i, data) for i in range(n_features)]
    # sort descending and take top m
    top_m = sorted(outlier_coefficients, reverse=True)[:m]
    proportion_of_outliers = (alpha * sum(top_m)) / float(m) if m > 0 else 0.0
    num_outliers = int(math.ceil(len(data) * proportion_of_outliers))

    if num_outliers <= 0:
        return ([], [])

    # find threshold for top num_outliers
    if num_outliers >= len(anomaly_scores):
        min_anomaly_score = min(anomaly_scores)
    else:
        min_anomaly_score = heapq.nlargest(num_outliers, anomaly_scores)[-1]

    outlier_candidates_indexes = [i for i in range(len(data)) if anomaly_scores[i] > min_anomaly_score]
    outlier_candidates = [data[i] for i in outlier_candidates_indexes]

    return outlier_candidates, outlier_candidates_indexes


def compute_lof(data: np.ndarray, candidates: List[np.ndarray], n_neighbors: int) -> np.ndarray:
    if len(candidates) == 0:
        return np.array([])
    lof = LocalOutlierFactor(n_neighbors=int(n_neighbors), novelty=True)
    lof.fit(data)
    return -lof.score_samples(np.asarray(candidates))


def continuous_scores(outlier_factors: np.ndarray, outlier_indexes: List[int], original_len: int) -> np.ndarray:
    res = np.zeros(original_len, dtype=float)
    for i, idx in enumerate(outlier_indexes):
        if 0 <= idx < original_len:
            res[idx] = float(outlier_factors[i])
    return res


class IFLOF(algorithmbase.AlgorithmBase):
    """IF-LOF pipeline wrapper.

    Hyperparameters (defaults inspired by the original script):
    - n_trees: number of isolation forest trees
    - max_samples: fraction or int for IsolationForest.max_samples
    - n_neighbors: LOF n_neighbors
    - alpha: multiplier used in pruning proportion calculation
    - m: number of attributes to consider for pruning (None -> all)
    - random_state: seed
    - contamination: optional thresholding in `predict` (default 0.1)
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "n_trees": 200,
            "max_samples": None,
            "n_neighbors": 20,
            "alpha": 1.0,
            "m": None,
            "random_state": 42,
            "contamination": 0.1,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self._last_scores: Optional[np.ndarray] = None
        # Keep the last pipeline artifacts for possible saving
        self._model = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults
            return
        merged = defaults.copy()
        merged.update(hyperparameter)
        # Use ParameterManagement to validate/complete types based on defaults
        try:
            pm = utils.ParameterManagement(lambda: None)
            pm._default_parameters = defaults.copy()
            validated = pm.check_hyperparameter_type(merged)
            merged = pm.complete_parameters(validated)
        except Exception:
            pass
        self._hyperparameter = merged

    def _run_pipeline(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X)
        hp = self._hyperparameter
        n_trees = int(hp.get("n_trees", 200))
        max_samples = hp.get("max_samples", None)
        n_neighbors = int(hp.get("n_neighbors", 20))
        alpha = float(hp.get("alpha", 1.0))
        m = hp.get("m", None)
        random_state = int(hp.get("random_state", 42))

        # IF scores
        if_scores = compute_iforest_scores(X, n_trees=n_trees, max_samples=max_samples, random_state=random_state)

        # prune
        candidates, candidate_indexes = prune_data(X, if_scores, alpha=alpha, m=m)

        # compute LOF on candidates
        lof_scores = compute_lof(X, candidates, n_neighbors=n_neighbors)

        # map back to continuous scores
        results = continuous_scores(lof_scores, candidate_indexes, original_len=len(X))

        # store
        self._last_scores = results
        # store minimal model for persistence
        self._model = {
            "hyperparameter": self._hyperparameter,
            "last_scores": self._last_scores,
        }
        return results

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        self.set_hyperparameter(self._hyperparameter)
        scores = self._run_pipeline(x_train)
        return scores

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        # if we've recently computed scores for this array, return them
        x = np.asarray(x)
        if self._last_scores is not None and len(self._last_scores) == len(x):
            return self._last_scores
        return self._run_pipeline(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.get_anomaly_score(x)
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        if not (0 < contamination < 0.5):
            contamination = max(0.001, min(0.5, contamination))
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        return (scores >= threshold).astype(int)

    def save_model(self, path_model: str):
        # Save the minimal _model using AlgorithmBase behaviour
        # ensure _model is present
        if self._model is None:
            self._model = {"hyperparameter": self._hyperparameter, "last_scores": self._last_scores}
        super().save_model(path_model)

    def load_model(self, path_model: str):
        super().load_model(path_model)
        # After loading, joblib will have set self._model
        # If the loaded object was a bare model (older behaviour), ignore
        try:
            self._model = self._model if self._model is not None else None
            if isinstance(self._model, dict):
                self._hyperparameter = self._model.get("hyperparameter", self._hyperparameter)
                self._last_scores = self._model.get("last_scores", self._last_scores)
        except Exception:
            pass
