from typing import Callable, List, Optional
import numpy as np
from scipy import sparse
from sklearn.linear_model import LinearRegression

from dantis.algorithmbase import AlgorithmBase


class Reservoir:
    def __init__(self, input_size: int, output_size: int, hidden_units: int,
                 connectivity: float, spectral_radius: float,
                 activation: Callable[[np.ndarray], np.ndarray]):
        self.hidden_units = hidden_units
        self.activation = activation
        self.W_in = np.random.uniform(-0.1, 0.1, (input_size, hidden_units))
        self.W_s = self._initialize_internal_weights(hidden_units, connectivity, spectral_radius)
        self.W_fb = np.random.uniform(-0.1, 0.1, (output_size, hidden_units))

    def _initialize_internal_weights(self, n_internal_units, connectivity, spectral_radius) -> np.ndarray:
        internal_weights = sparse.rand(n_internal_units, n_internal_units, density=connectivity).todense()
        internal_weights[np.where(internal_weights > 0)] -= 0.5
        # adjust spectral radius
        E, _ = np.linalg.eig(internal_weights)
        e_max = np.max(np.abs(E))
        if e_max == 0:
            return internal_weights
        internal_weights /= np.abs(e_max) / spectral_radius
        return internal_weights

    def _calc_state(self, x: np.ndarray, last_state: np.ndarray, last_output: np.ndarray):
        state = x.dot(self.W_in) + last_state.dot(self.W_s) + last_output.dot(self.W_fb)
        state = self.activation(state)
        return state

    def fit_transform(self, X_tuple, y=None, **_):
        current_input, last_state, last_output = X_tuple
        if last_state is None and last_output is None:
            last_state = np.zeros((1, self.hidden_units))
            last_output = np.zeros_like(current_input)
        state = self._calc_state(current_input, last_state, last_output)
        return state


class HealthESN:
    """Modelo ESN para detección de anomalías.

    Interfaz reducida de la implementación original. Esta clase expone:
      - fit(X) -> entrena w_out (regresión lineal)
      - predict(X) -> retorna un array 1D con puntuaciones de anomalía
    """
    def __init__(self,
                 n_dimensions: int,
                 hidden_units: int = 500,
                 window_size: int = 20,
                 connectivity: float = 0.25,
                 spectral_radius: float = 0.6,
                 activation: Callable[[np.ndarray], np.ndarray] = np.tanh,
                 seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(int(seed))

        self.esn = Reservoir(n_dimensions, n_dimensions, hidden_units, connectivity, spectral_radius, activation)
        self.w_out = LinearRegression()
        self.window_size = int(window_size)
        sigma = np.arange(self.window_size)[::-1]
        self.sigma = sigma / sigma.sum()

    def fit(self, X: np.ndarray) -> 'HealthESN':
        # X expected shape (T, D)
        y = X[1:]
        x = X[:-1]

        last_state = None
        last_output = None
        states: List[np.ndarray] = []
        for t in range(x.shape[0]):
            x_ = (x[[t]], last_state, last_output)
            state = self.esn.fit_transform(x_)
            states.append(state)
            last_state = state
            last_output = y[[t]]

        self.w_out.fit(np.concatenate(states, axis=0), y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        states = []
        last_state = None
        last_output = None
        # start producing states after window_size
        for i in range(self.window_size, X.shape[0]):
            for p in reversed(range(1, self.window_size + 1)):
                x = (X[i - p], last_state, last_output)
                state = self.esn.fit_transform(x)
                last_state = state
                last_output = X[i - p + 1]
            states.append(last_state)
        if len(states) == 0:
            # not enough data to compute scores
            return np.full(X.shape[0], np.nan)

        outputs = self.w_out.predict(np.concatenate(states, axis=0))
        scores = np.linalg.norm(X[self.window_size:] - outputs, axis=1)
        # pad with nan at beginning to keep same length
        pad = np.full(X.shape[0] - scores.shape[0], np.nan)
        scores = np.concatenate([pad, scores])
        return scores


class HealthESNAlgorithm(AlgorithmBase):
    """Wrapper que adapta HealthESN a AlgorithmBase.

    Hyperparameters aceptados (con sus valores por defecto):
      - linear_hidden_size: 500
      - prediction_window_size: 20
      - connectivity: 0.25
      - spectral_radius: 0.6
      - activation: 'tanh' or 'sigmoid'
      - random_state: 42

    Uso:
      alg = HealthESNAlgorithm({ ... })
      alg.fit(X_train)
      scores = alg.decision_function(X_test)
      labels = alg.predict(X_test)
    """

    def __init__(self, hyperparameter: Optional[dict] = None) -> None:
        defaults = {
            "linear_hidden_size": 500,
            "prediction_window_size": 20,
            "connectivity": 0.25,
            "spectral_radius": 0.6,
            "activation": "tanh",
            "random_state": 42,
        }
        merged = defaults.copy()
        if hyperparameter:
            merged.update(hyperparameter)
        super().__init__(merged)
        self._model: Optional[HealthESN] = None

    def _get_activation(self, name: str) -> Callable[[np.ndarray], np.ndarray]:
        name = (name or "tanh").lower()
        if name == "sigmoid":
            from scipy.special import expit
            return expit
        return np.tanh

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        x = np.asarray(x_train)
        params = self.get_hyperparameter() or {}
        activation = self._get_activation(params.get("activation"))
        model = HealthESN(
            n_dimensions=x.shape[1] if x.ndim == 2 else 1,
            hidden_units=int(params.get("linear_hidden_size", 500)),
            window_size=int(params.get("prediction_window_size", 20)),
            connectivity=float(params.get("connectivity", 0.25)),
            spectral_radius=float(params.get("spectral_radius", 0.6)),
            activation=activation,
            seed=int(params.get("random_state", 42))
        )
        model.fit(x)
        self._model = model
        return self

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Model not trained. Call fit() before decision_function().")
        x = np.asarray(x)
        # ensure 2D (T, D)
        if x.ndim == 1:
            x = x.reshape(-1, 1)
        return self._model.predict(x)

    def predict(self, x: np.ndarray) -> np.ndarray:
        # simple binary prediction: anomaly if score is not nan and > 0
        scores = self.decision_function(x)
        # treat nan as 0 (not enough data)
        labels = (~np.isnan(scores)) & (scores > 0)
        return labels.astype(int)
