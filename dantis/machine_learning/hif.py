import numpy as np
import random as rn
from collections import Counter
from typing import Optional, Dict, Any

import logging

from .. import algorithmbase
from .. import utils

logger = logging.getLogger(__name__)


def EuclideanDist(x, y):
    return np.sqrt(np.sum((x - y) ** 2))


def c_factor(n):
    if n < 2:
        n = 2
    return 2.0 * (np.log(n - 1) + 0.5772156649) - (2.0 * (n - 1.0) / (n * 1.0))


class hiForest(object):
    def buildTree(self, i):
        ix = rn.sample(range(self.nobjs), self.sample)
        X_p = self.X[ix]
        return hiTree(X_p, 0, self.limit)

    def __init__(self, X, ntrees, sample_size, limit=None, nCore=1):
        self.ntrees = ntrees
        self.X = X
        self.nobjs = len(X)
        self.sample = sample_size
        self.Trees = []
        self.limit = limit
        self.nCore = nCore
        self.X_in = None

        if limit is None:
            self.limit = int(np.ceil(1.2 * np.log2(self.sample)))
        self.c = c_factor(self.sample)
        self.Trees = []
        for i in range(self.ntrees):
            ix = rn.sample(range(self.nobjs), self.sample)
            X_p = X[ix]
            self.Trees.append(hiTree(X_p, 0, self.limit))

    def computeScore_paths(self, X_in=None):
        if X_in is None:
            X_in = self.X
        S = np.zeros(len(X_in))
        for i in range(len(X_in)):
            h_temp = 0
            for j in range(self.ntrees):
                h_temp += PathFactor(X_in[i], self.Trees[j]).path * 1.0
            Eh = h_temp / self.ntrees
            S[i] = 2.0 ** (-Eh / self.c)
        return S

    def computeScore(self, i):
        h_temp = 0
        for j in range(self.ntrees):
            h_temp += PathFactor(self.X_in[i], self.Trees[j]).path * 1.0
        Eh = h_temp / self.ntrees
        return 2.0 ** (-Eh / self.c)

    def computeScore_pathsPool(self, X_in=None):
        from multiprocessing import Pool

        pool = Pool(self.nCore)
        if X_in is None:
            X_in = self.X
        self.X_in = X_in
        L = len(X_in)
        tab = list(range(L))
        S = pool.map(self.computeScore, tab)
        return S

    def computeScore_paths_single(self, x):
        S = np.zeros(self.ntrees)
        for j in range(self.ntrees):
            path = PathFactor(x, self.Trees[j]).path * 1.0
            S[j] = 2.0 ** (-1.0 * path / self.c)
        return S

    def computeScore_paths_single_with_labs(self, x):
        S = np.zeros(self.ntrees)
        labs = []
        for j in range(self.ntrees):
            pf = PathFactor(x, self.Trees[j])
            path = pf.path * 1.0
            S[j] = 2.0 ** (-1.0 * path / self.c)
            labs.append(pf.labs)
        return S, labs

    def computeAggScore(self, x):
        S = np.zeros(self.ntrees)
        labsCount = Counter([])
        ldist = []
        ldist_a = []
        for j in range(self.ntrees):
            pf = PathFactor(x, self.Trees[j])
            path = pf.path * 1.0
            S[j] = 2.0 ** (-1.0 * path / self.c)
            labsCount = labsCount + pf.labs
            if len(pf.ldist) > 0:
                ldist.append(np.mean(pf.ldist))
            if len(pf.ldist_a) > 0:
                ldist_a.append(np.mean(pf.ldist_a, axis=0))
        meanDist = 0
        if len(ldist) > 0:
            meanDist = np.mean(ldist)
        meanDist_r = 0
        if len(ldist_a) > 0:
            meanDist_a = np.mean(ldist_a, axis=0)
            if meanDist_a > 0:
                meanDist_r = meanDist / (meanDist_a)
        return np.mean(S), labsCount, meanDist, meanDist_r

    def addAnomaly(self, x, lab):
        for j in range(self.ntrees):
            pf = PathFactor(x, self.Trees[j])
            pf.addAnomaly(x, lab, self.Trees[j].root)

    def computeAnomalyCentroid(self):
        for j in range(self.ntrees):
            self.Trees[j].root.computeAnomalyCentroid()

    def getAverageBucketSize(self):
        szb = 0
        nbb = 0
        for j in range(self.ntrees):
            s, n = self.Trees[j].root.getAverageBucketSize()
            szb += s
            nbb += n
        out = 0
        if nbb > 0:
            out = szb / nbb
        return out, nbb / self.ntrees


class Node(object):
    def __init__(self, X, q, p, e, left, right, node_type=""):
        self.e = e
        self.size = len(X)
        self.X = X
        self.q = q
        self.p = p
        self.left = left
        self.right = right
        self.ntype = node_type
        self.C = None
        self.Ca = None
        self.labs = []
        self.Xanomaly = []
        if node_type == "exNode" and self.size > 0:
            self.C = np.mean(X, axis=0)

    def computeAnomalyCentroid(self):
        if self.ntype == "exNode":
            if len(self.Xanomaly) > 0:
                self.Ca = np.mean(self.Xanomaly, axis=0)
        else:
            self.left.computeAnomalyCentroid()
            self.right.computeAnomalyCentroid()

    def getAverageBucketSize(self):
        if self.ntype == "exNode":
            return self.size, 1
        else:
            s1, n1 = self.left.getAverageBucketSize()
            s2, n2 = self.right.getAverageBucketSize()
            return s1 + s2, n1 + n2


class hiTree(object):
    def __init__(self, X, e, l):
        self.e = e
        self.X = X
        self.size = len(X)
        self.Q = np.arange(np.shape(X)[1], dtype="int")
        self.l = l
        self.p = None
        self.q = None
        self.exnodes = 0
        self.labs = []
        self.root = self.make_tree(X, e, l)

    def make_tree(self, X, e, l):
        self.e = e
        if e >= l or len(X) <= 1:
            left = None
            right = None
            self.exnodes += 1
            return Node(X, self.q, self.p, e, left, right, node_type="exNode")
        else:
            self.q = rn.choice(self.Q)
            self.p = rn.uniform(X[:, self.q].min(), X[:, self.q].max())
            w = np.where(X[:, self.q] < self.p, True, False)
            return Node(
                X,
                self.q,
                self.p,
                e,
                left=self.make_tree(X[w], e + 1, l),
                right=self.make_tree(X[~w], e + 1, l),
                node_type="inNode",
            )

    def get_node(self, path):
        node = self.root
        for p in path:
            if p == "L":
                node = node.left
            if p == "R":
                node = node.right
        return node


class PathFactor(object):
    def __init__(self, x, hitree):
        self.path_list = []
        self.labs = []
        self.ldist = []
        self.ldist_a = []
        self.x = x
        self.e = 0
        self.path = self.find_path(hitree.root)

    def find_path(self, T):
        if T.ntype == "exNode":
            self.labs = Counter(T.labs)
            if not (T.C is None):
                self.ldist.append(EuclideanDist(self.x, T.C))
            if not (T.Ca is None):
                self.ldist_a.append(EuclideanDist(self.x, T.Ca))
            sz = T.size
            if sz == 0:
                sz += 1
            for key in self.labs:
                self.labs[key] /= sz
            if T.size == 1:
                return self.e
            else:
                self.e = self.e + c_factor(T.size)
                return self.e
        else:
            a = T.q
            self.e += 1
            if self.x[a] < T.p:
                self.path_list.append("L")
                return self.find_path(T.left)
            else:
                self.path_list.append("R")
                return self.find_path(T.right)

    def addAnomaly(self, x, lab, T):
        if T.ntype == "exNode":
            T.labs.append(lab)
            T.Xanomaly.append(x)
        else:
            a = T.q
            if self.x[a] < T.p:
                return self.addAnomaly(x, lab, T.left)
            else:
                return self.addAnomaly(x, lab, T.right)


class HIF(algorithmbase.AlgorithmBase):
    """
    Wrapper for the hiForest algorithm.

    Hyperparameters (defaults available via `get_default_hyperparameters`):
    - n_trees: number of trees
    - max_samples: either fraction (0-1) of samples to use per tree or None (will use min(256, n_samples))
    - random_state: seed
    - limit: depth limit for trees (None -> computed)
    - n_core: number of cores for parallel scoring (not used by default)
    - contamination: used to convert scores -> binary labels in `predict`
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "n_trees": 1024,
            "max_samples": None,
            "random_state": 42,
            "limit": None,
            "n_core": 1,
            "contamination": 0.1,
        }

    def __init__(self, hyperparameter: Optional[Dict[str, Any]] = None):
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        # model will be created during fit because hiForest requires training data in constructor
        self._model = None

    def set_hyperparameter(self, hyperparameter: Optional[Dict[str, Any]]):
        defaults = self.get_default_hyperparameters()
        if hyperparameter is None:
            self._hyperparameter = defaults
            return
        # merge provided hyperparameters onto defaults
        merged = defaults.copy()
        merged.update(hyperparameter)
        # optional: use ParameterManagement to check types when possible
        try:
            parameters_management = utils.ParameterManagement(hiForest.__init__)
            validated = parameters_management.check_hyperparameter_type(merged)
            merged = parameters_management.complete_parameters(validated)
        except Exception:
            # ParameterManagement may not provide useful defaults for hiForest signature
            pass
        self._hyperparameter = merged

    def _set_seed(self):
        rs = self._hyperparameter.get("random_state", None)
        if rs is not None:
            import random

            random.seed(rs)
            rn.seed(rs)
            np.random.seed(rs)

    def _compute_sample_size(self, n_samples: int) -> int:
        max_samples = self._hyperparameter.get("max_samples", None)
        if max_samples:
            # if float assume fraction
            if isinstance(max_samples, float) and 0 < max_samples <= 1:
                sample_size = int(max_samples * n_samples)
            else:
                sample_size = int(max_samples)
        else:
            sample_size = min(256, n_samples)
        sample_size = max(2, sample_size)
        return sample_size

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """Build the hiForest using x_train. If y_train is provided, it will be used
        to separate normal points from anomalies and anomalies will be added to
        the tree buckets (matching TimeEval behaviour).

        Returns anomaly scores for x_train (np.ndarray, shape (n_samples,)).
        """
        self.set_hyperparameter(self._hyperparameter)
        self._set_seed()

        X = np.asarray(x_train)
        n_samples = X.shape[0]
        sample_size = self._compute_sample_size(n_samples)
        n_trees = int(self._hyperparameter.get("n_trees", 1024))

        # If y_train given, separate normals and anomalies
        train_data = X
        anomalies = None
        if y_train is not None:
            y = np.asarray(y_train).astype(bool)
            if y.shape[0] == n_samples:
                train_data = X[~y]
                anomalies = X[y]

        # ensure there is at least one sample
        if train_data.shape[0] == 0:
            raise ValueError("No normal samples available to build hiForest")

        # instantiate model
        limit = self._hyperparameter.get("limit", None)
        n_core = int(self._hyperparameter.get("n_core", 1))
        self._model = hiForest(train_data, n_trees, sample_size, limit=limit, nCore=n_core)

        # add anomalies (if any) and compute centroids
        if anomalies is not None and anomalies.shape[0] > 0:
            for i in range(anomalies.shape[0]):
                self._model.addAnomaly(x=anomalies[i], lab=1)
            self._model.computeAnomalyCentroid()

        # compute and return scores for the full x_train (including anomalies if provided)
        scores = np.zeros(n_samples)
        for i in range(n_samples):
            score, *_ = self._model.computeAggScore(X[i])
            scores[i] = score

        return scores

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """Return anomaly scores for input samples (higher -> more anomalous)."""
        if self._model is None:
            raise RuntimeError("Model has not been fitted. Call `fit` first.")

        X = np.asarray(x)
        n = X.shape[0]
        scores = np.zeros(n)
        for i in range(n):
            score, *_ = self._model.computeAggScore(X[i])
            scores[i] = score
        return scores

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return binary labels (1 anomaly, 0 normal) using the `contamination` percentile."""
        scores = self.get_anomaly_score(x)
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        if not (0 < contamination < 0.5):
            # clamp to reasonable range
            contamination = max(0.001, min(0.5, contamination))
        # higher scores -> more anomalous. Use top contamination fraction as anomalies
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        return (scores >= threshold).astype(int)

    def save_model(self, path_model: str):
        super().save_model(path_model)

    def load_model(self, path_model: str):
        super().load_model(path_model)
