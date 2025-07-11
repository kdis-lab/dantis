import numpy as np
import logging
from .. import algorithmbase
from .. import utils
from . import topology, layers

logging.basicConfig(level=logging.INFO)


class AELSTM(algorithmbase.AlgorithmBase):
    """
    AELSTM(hyperparameter: dict)

    LSTM-based Autoencoder for time series anomaly detection.

    This class implements an LSTM Autoencoder architecture for detecting anomalies in time series data. The model is composed of LSTM layers for encoding and decoding the input sequence, with a RepeatVector layer to reconstruct the sequence and a TimeDistributed Dense layer for output. It inherits from `algorithmbase.AlgorithmBase` and provides methods for model training, prediction, and anomaly scoring.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model and training hyperparameters. See `get_default_hyperparameters` for defaults.

    Attributes
    ----------
    _model : object
        The underlying LSTM autoencoder model instance.
    _hyperparameter : dict
        Dictionary of hyperparameters for the model.

    Methods
    -------
    get_default_hyperparameters() -> dict
        Returns a dictionary of default hyperparameters for the model.

    decision_function(x)
        Computes the reconstruction error for the input data `x` using the trained autoencoder.

    fit(x_train: np.array, y_train: np.array = None)
        Trains the LSTM autoencoder model on the provided training data.

    predict(x: np.array = None)
        Returns the anomaly score for the input data `x`.

    save_model(path_model)
        Saves the trained model to the specified path.

    load_model(path_model)
        Loads a model from the specified path.

    set_hyperparameter(hyperparameter)
        Sets the hyperparameters for the model.

    _create_model()
        Internal method to build the LSTM autoencoder architecture based on the current hyperparameters.

    Notes
    -----
    - The model expects input data of shape `(n_samples, sequence_length, n_features)`.
    - The anomaly score is typically based on the reconstruction error of the autoencoder.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the AELSTM model with the given hyperparameters.

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
        Trains the AELSTM model with the training data.

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
        self._hyperparameter.update(default_hyperparams)
        input_shape = self._hyperparameter["input_shape"]
        units = self._hyperparameter.get("units", 64)
        optimizer = self._hyperparameter.get("optimizer", "adam")
        loss = self._hyperparameter.get("loss", "mse")
        seq_len, n_feat = input_shape

        layer_list = [
            layers.Layer(layers.LayerType.LSTM, {"units": units, "return_sequences": False}).layer,
            layers.Layer(layers.LayerType.REPEAT_VECTOR, {"n": seq_len}).layer,
            layers.Layer(layers.LayerType.LSTM, {"units": units, "return_sequences": True}).layer,
            layers.Layer(layers.LayerType.TIME_DISTRIBUTED, {"layer": "Dense", "units": n_feat}).layer
        ]

        hparam = {
            "input_shape": input_shape,
            "layers": layer_list,
            "compile": {"optimizer": optimizer, "loss": loss},
            "fit": {"x": None, "y": None}
        }

        self._model = topology.Topology(hparam)
