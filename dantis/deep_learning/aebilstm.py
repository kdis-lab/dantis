import numpy as np
import logging
from .. import algorithmbase
from .. import utils
from . import topology, layers

logging.basicConfig(level=logging.INFO)


class AEBiLSTM(algorithmbase.AlgorithmBase):
    """
    Autoencoder Bidirectional LSTM (AEBiLSTM) for anomaly detection in time series.

    This class implements an autoencoder based on Bidirectional LSTM layers, designed to learn representations
    of temporal sequences and detect anomalies through reconstruction error.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary of hyperparameters for model configuration. May include:
            - input_shape: tuple
                Input shape as (sequence length, number of features).
            - units: int
                Number of LSTM units.
            - optimizer: str
                Optimizer for model compilation.
            - loss: str
                Loss function.
            - epochs: int
                Number of training epochs.
            - batch_size: int
                Batch size.
            - verbose: int
                Verbosity level.

    Attributes
    ----------
    _model : object
        The underlying autoencoder model.
    _hyperparameter : dict
        Dictionary of model hyperparameters.

    Methods
    -------
    get_default_hyperparameters() -> dict
        Returns a dictionary with the default model hyperparameters.

    decision_function(x)
        Computes the reconstruction error for input samples `x`.

    fit(x_train, y_train=None)
        Trains the autoencoder model with the training data.

    predict(x=None)
        Computes the anomaly score for input samples `x`.

    save_model(path_model)
        Saves the model to the specified path.

    load_model(path_model)
        Loads the model from the specified path.

    set_hyperparameter(hyperparameter)
        Sets the model hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the AEBiLSTM model with the given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of model hyperparameters.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._model = None

    @staticmethod
    def get_default_hyperparameters() -> dict:
        """
        Returns the default hyperparameters for the model.

        Returns
        -------
        dict
            Default hyperparameters.
        """
        return {
            "input_shape": (10, 1),
            "units": 64,
            "optimizer": "adam",
            "loss": "mse",
            "epochs": 50,
            "batch_size": 32,
            "verbose": 0
        }

    def decision_function(self, x):
        """
        Computes the reconstruction error for input samples `x`.

        Parameters
        ----------
        x : np.array
            Input data.

        Returns
        -------
        np.array
            Reconstruction error for each sample.
        """
        return self._model.predict_reconstruction_error(x)

    def fit(self, x_train: np.array, y_train: np.array = None):
        """
        Trains the autoencoder model with the training data.

        Parameters
        ----------
        x_train : np.array
            Training data.
        y_train : np.array, optional
            Target data (not used).

        Returns
        -------
        np.array
            Anomaly scores for the training data.
        """
        self._create_model()
        self._model.fit(x_train, y_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.array = None):
        """
        Computes the anomaly score for input samples `x`.

        Parameters
        ----------
        x : np.array, optional
            Input data.

        Returns
        -------
        np.array
            Anomaly scores.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path_model : str
            Path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path_model : str
            Path to load the model from.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Sets the model hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter

    def _create_model(self):
        """
        Internal method to build the autoencoder model architecture.
        """
        default_hyperparams = self.get_default_hyperparameters()
        self._hyperparameter = default_hyperparams.update(self._hyperparameter)
        input_shape = self._hyperparameter.get("input_shape", (2, 1))
        units = self._hyperparameter.get("units", 64)
        optimizer = self._hyperparameter.get("optimizer", "adam")
        loss = self._hyperparameter.get("loss", "mse")
        seq_len, n_feat = input_shape
        inner_layer = layers.Layer(layers.LayerType.DENSE, {"units": n_feat}).layer
        layer_list = [
            layers.Layer(layers.LayerType.BIDIRECTIONAL_LSTM, {"units": units, "return_sequences": False}).layer,
            # layers.Layer(layers.LayerType.REPEAT_VECTOR, {"n": seq_len}).layer, # Posible capa de mejora TODO faltaria añadirla
            layers.Layer(layers.LayerType.BIDIRECTIONAL_LSTM, {"units": units, "return_sequences": True}).layer,
            layers.Layer(layers.LayerType.TIMEDISTRIBUTED, {"layer": inner_layer}).layer
        ]

        hparam = {
            "input_shape": input_shape,
            "layers": layer_list,
            "compile": {"optimizer": optimizer, "loss": loss},
            "fit": {"x": None, "y": None}
        }

        self._model = topology.Topology(hparam)
