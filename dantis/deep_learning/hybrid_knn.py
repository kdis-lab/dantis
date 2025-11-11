import os
from typing import List, Optional, Dict, Any, Callable
import logging

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.neighbors import NearestNeighbors

from .. import algorithmbase
from .. import utils

logger = logging.getLogger(__name__)


class Activation(str):
    RELU = "relu"
    SIGMOID = "sigmoid"

    def to_torch(self):
        if self == Activation.RELU:
            return nn.ReLU()
        return nn.Sigmoid()


class TimeSeriesDataset(Dataset):
    def __init__(self, X: np.ndarray, window_size: int):
        self.X = torch.from_numpy(X).float()
        self.window_size = window_size

    def __len__(self):
        return self.X.shape[0] - (self.window_size - 1)

    def __getitem__(self, index):
        end_idx = index + self.window_size
        x = self.X[index:end_idx].reshape(-1)
        return x


class EarlyStopping:
    def __init__(self, patience: int, delta: float, epochs: int,
                 callbacks: Optional[List[Callable]] = None):
        self.patience = patience
        self.delta = delta
        self.epochs = epochs
        self.current_epoch = 0
        self.epochs_without_change = 0
        self.last_loss = None
        self.callbacks = callbacks or []

    def _callback(self, improvement: bool, loss: float):
        for cb in self.callbacks:
            try:
                cb(improvement, loss, self.epochs_without_change)
            except Exception:
                logger.exception("EarlyStopping callback failed")

    def update(self, loss: float):
        improvement = False
        if self.last_loss is None or (1 - (loss / self.last_loss) > self.delta):
            self.last_loss = loss
            self.epochs_without_change = 0
            improvement = True
        else:
            self.epochs_without_change += 1

        self._callback(improvement, loss)

    def __iter__(self):
        while self.epochs_without_change <= self.patience and self.current_epoch < self.epochs:
            yield self.current_epoch
            self.current_epoch += 1


class Encoder(nn.Module):
    def __init__(self, layer_sizes: List[int], activation: Activation):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ])
        self.activation = activation.to_torch()

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = self.activation(layer(x))
        return x


class Decoder(nn.Module):
    def __init__(self, layer_sizes: List[int]):
        super().__init__()
        # reversed mapping
        self.layers = nn.ModuleList([
            nn.Linear(layer_sizes[i], layer_sizes[i - 1]) for i in reversed(range(1, len(layer_sizes)))
        ])

    def forward(self, x: torch.Tensor):
        for layer in self.layers:
            x = layer(x)
        return x


class DAE(nn.Module):
    def __init__(self,
                 layer_sizes: List[int],
                 activation: Activation,
                 split: float,
                 window_size: int,
                 batch_size: int,
                 test_batch_size: int,
                 epochs: int,
                 early_stopping_delta: float,
                 early_stopping_patience: int,
                 learning_rate: float):
        super().__init__()
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.split = split
        self.window_size = window_size
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.epochs = epochs
        self.early_stopping_delta = early_stopping_delta
        self.early_stopping_patience = early_stopping_patience
        self.lr = learning_rate

        self.encoder = Encoder(layer_sizes, self.activation)
        self.decoder = Decoder(layer_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def fit(self, ts: np.ndarray, callbacks: Optional[List[Callable]] = None, verbose: bool = True):
        self.train()
        logger_da = logging.getLogger("DAE")
        optimizer = Adam(self.parameters(), lr=self.lr)
        criterion = nn.MSELoss()

        train_dl, valid_dl = self._split_data(ts)
        early_stopping = EarlyStopping(self.early_stopping_patience, self.early_stopping_delta, self.epochs,
                                       callbacks=callbacks)

        for epoch in early_stopping:
            self.train()
            losses = []
            for x in train_dl:
                optimizer.zero_grad()
                loss = self._predict(x, criterion)
                loss.backward()
                optimizer.step()
                losses.append(loss.item())

            self.eval()
            valid_losses = []
            for x in valid_dl:
                loss = self._predict(x, criterion)
                valid_losses.append(loss.item())
            validation_loss = sum(valid_losses)
            early_stopping.update(validation_loss)
            if verbose:
                logger_da.info(
                    f"Epoch {epoch}: Training Loss {sum(losses) / len(train_dl):.6f} \t "
                    f"Validation Loss {validation_loss / len(valid_dl):.6f}"
                )

    def _predict(self, x, criterion) -> torch.Tensor:
        y_hat = self.forward(x)
        loss = criterion(y_hat, x)
        return loss

    def reduce(self, X: np.ndarray) -> np.ndarray:
        self.eval()
        dl = DataLoader(TimeSeriesDataset(X, window_size=self.window_size), batch_size=self.test_batch_size)
        reduced_x = []
        for x in dl:
            reduced_x.append(self.encoder(x))
        return torch.cat(reduced_x).detach().numpy()

    def _split_data(self, ts: np.ndarray):
        split_at = int(len(ts) * self.split)
        train_ts = ts[:split_at]
        valid_ts = ts[split_at:]
        train_ds = TimeSeriesDataset(train_ts, window_size=self.window_size)
        valid_ds = TimeSeriesDataset(valid_ts, window_size=self.window_size)
        return DataLoader(train_ds, batch_size=self.batch_size), DataLoader(valid_ds, batch_size=self.test_batch_size)


class KNN:
    def __init__(self, k: int):
        self.k = k
        self.nbrs = NearestNeighbors()

    def fit(self, X: np.ndarray):
        self.nbrs.fit(X)
        return self

    def kth_neighbor_distance(self, X: np.ndarray):
        distances, _ = self.nbrs.kneighbors(X, self.k, return_distance=True)
        return distances[:, -1]


class KNNEnsemble:
    def __init__(self, k: int, l: int):
        self.k = k
        self.l = l
        self.s = []
        self.g = []
        self.knn_models = []

    def fit(self, X: np.ndarray):
        self._eventually_reduce_estimator_size(X.shape)
        X_sh = X.copy()
        np.random.shuffle(X_sh)
        self.s = np.array_split(X_sh, self.l)
        if not self._validate_fitting():
            self._init_knn_models()
        for knn, s in zip(self.knn_models, self.s):
            knn.fit(s)
        self.g = [self._calculate_kth_distance(s) for s in self.s]
        return self

    def _init_knn_models(self):
        self.knn_models = [KNN(self.k) for _ in range(self.l)]

    def _calculate_kth_distance(self, X: np.ndarray):
        if not self._validate_fitting():
            raise ValueError("KNNEnsemble not fitted yet")
        d = np.zeros((X.shape[0], self.l))
        for l, knn in enumerate(self.knn_models):
            d[:, l] = knn.kth_neighbor_distance(X)
        g = d.mean(axis=1)
        return g

    def _validate_fitting(self):
        return len(self.knn_models) > 0

    def _eventually_reduce_estimator_size(self, data_shape):
        max_estimators = data_shape[0] // self.k
        if max_estimators < self.l:
            logging.warning(f"The dataset is too small for the number of estimators ({self.l}). We set `n_estimators` to {max_estimators}!")
        self.l = min(max_estimators, self.l)

    def predict(self, X: np.ndarray):
        g = self._calculate_kth_distance(X)
        p = np.zeros((X.shape[0], self.l))
        for l in range(self.l):
            identity = np.greater(g.reshape(-1, 1), self.g[l].reshape(1, -1)).astype(int)
            p[:, l] = identity.mean(axis=1)
        return p.mean(axis=1)


class HybridKNN(algorithmbase.AlgorithmBase):
    """Wrapper that replicates TimeEval's HybridKNN behavior inside dantis.

    Hyperparameters mirror the TimeEval `CustomParameters` for hybrid_knn.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        return {
            "linear_layer_shape": [100, 10],
            "activation": "relu",
            "split": 0.8,
            "anomaly_window_size": 20,
            "batch_size": 64,
            "test_batch_size": 256,
            "epochs": 1,
            "early_stopping_delta": 0.05,
            "early_stopping_patience": 10,
            "learning_rate": 0.001,
            "n_neighbors": 12,
            "n_estimators": 3,
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
        # Use ParameterManagement to validate/complete types based on defaults
        try:
            pm = utils.ParameterManagement(lambda: None)
            pm._default_parameters = defaults.copy()
            validated = pm.check_hyperparameter_type(merged)
            merged = pm.complete_parameters(validated)
        except Exception:
            pass
        self._hyperparameter = merged

    def _set_seed(self):
        rs = self._hyperparameter.get("random_state", None)
        if rs is not None:
            import random
            random.seed(rs)
            np.random.seed(rs)
            torch.manual_seed(rs)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """Train the DAE and KNN ensemble. Returns anomaly scores for x_train."""
        self.set_hyperparameter(self._hyperparameter)
        self._set_seed()

        X = np.asarray(x_train)
        n_features = X.shape[1]
        window = int(self._hyperparameter.get("anomaly_window_size", 20))
        input_size = n_features * window
        linear_shape = list(self._hyperparameter.get("linear_layer_shape", [100, 10]))
        layer_sizes = [input_size] + linear_shape
        activation = Activation(self._hyperparameter.get("activation", "relu"))

        # instantiate components
        dae = DAE(layer_sizes, activation, float(self._hyperparameter.get("split", 0.8)), window,
                  int(self._hyperparameter.get("batch_size", 64)), int(self._hyperparameter.get("test_batch_size", 256)),
                  int(self._hyperparameter.get("epochs", 1)), float(self._hyperparameter.get("early_stopping_delta", 0.05)),
                  int(self._hyperparameter.get("early_stopping_patience", 10)), float(self._hyperparameter.get("learning_rate", 0.001)))

        knn = KNNEnsemble(int(self._hyperparameter.get("n_neighbors", 12)), int(self._hyperparameter.get("n_estimators", 3)))

        # simple training: train DAE then fit KNN on reduced space
        dae.fit(X, callbacks=None, verbose=True)
        reduced_X = dae.reduce(X)
        knn.fit(reduced_X)

        # store model tuple
        self._model = {"dae": dae, "knn": knn}

        # return training anomaly scores
        scores = knn.predict(reduced_X)
        return scores

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not fitted. Call `fit` first.")
        dae = self._model["dae"]
        knn = self._model["knn"]
        reduced = dae.reduce(np.asarray(x))
        return knn.predict(reduced)

    def predict(self, x: np.ndarray) -> np.ndarray:
        scores = self.get_anomaly_score(x)
        contamination = float(self._hyperparameter.get("contamination", 0.1))
        if not (0 < contamination < 0.5):
            contamination = max(0.001, min(0.5, contamination))
        threshold = np.percentile(scores, 100.0 * (1.0 - contamination))
        return (scores >= threshold).astype(int)

    def save_model(self, path_model: str):
        if self._model is None:
            raise RuntimeError("No model to save")
        # save with torch to preserve modules
        torch.save(self._model, path_model)

    def load_model(self, path_model: str):
        self._model = torch.load(path_model)
