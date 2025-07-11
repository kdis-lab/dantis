from pyod.models import vae # vae.VAE
import numpy as np
import joblib
import keras

from .. import algorithmbase
from .. import utils


class VAE(algorithmbase.AlgorithmBase):
    """
    VAE(hyperparameter: dict)
    Variational Autoencoder (VAE) for unsupervised anomaly detection.
    This class implements the VAE algorithm, a probabilistic generative model used for detecting anomalies in data. 
    It inherits from `algorithmbase.AlgorithmBase` and provides methods for model training, prediction, and anomaly scoring.
    The VAE model learns a latent representation of the input data and identifies anomalies based on reconstruction error or likelihood.
    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model and training hyperparameters. See `get_default_hyperparameters` for defaults.
    Attributes
    ----------
    _model : object
        The underlying VAE model instance.
    _hyperparameter : dict
        Dictionary of hyperparameters for the model.
    Methods
    -------
    decision_function(x)
        Computes the anomaly score for the input data `x` using the trained VAE model.
    fit(x_train: np.ndarray, y_train: np.ndarray = None)
        Trains the VAE model on the provided training data.
    predict(x: np.ndarray = None) -> np.ndarray
        Returns the anomaly score for the input data `x`.
    save_model(path_model)
        Saves the trained VAE model to the specified path.
    load_model(path_model)
        Loads a VAE model from the specified path.
    set_hyperparameter(hyperparameter)
        Sets the hyperparameters for the VAE model.
    _create_model()
        Internal method to build the VAE model based on the current hyperparameters.
    Notes
    -----
    - The model is designed for unsupervised anomaly detection and does not use `y_train` during training.
    - The anomaly score is typically based on the reconstruction error or the likelihood under the learned model.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the VAE model with the given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of model hyperparameters.
        """
        super().__init__(hyperparameter=hyperparameter)
        
        self._create_model()

    def decision_function(self, x):
        """
        Predict raw anomaly score of X using the fitted detector. 
        The anomaly score of an input sample is computed based on different detector algorithms. 
        For consistency, outliers are assigned with larger anomaly scores.
        Parameters
        ----------
        x : np.array
            Input data.

        Returns
        -------
        np.array
            The anomaly score of the input samples.        
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.array, y_train: np.array = None) -> np.array:
        """
        Trains the vae model with the training data.

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

        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.array = None) -> np.array:
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
        extension = "extension"
        if "." in path_model:
            extension = path_model[path_model.rfind(".")+1:]
            if extension != "joblib":
                print("Warning: Change extension")
                extension = "joblib"
        self._model.model_.save("model")
        self._model.model_ = None
        path_model = path_model[:path_model.rfind(".")+1] + extension
        joblib.dump(self._model, path_model)

    def load_model(self, path_model):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path_model : str
            Path to load the model from.
        """
        if "." in path_model:
            extension = path_model[path_model.rfind(".")+1:]
            if extension != "joblib":
                raise (Exception("Error: extension required .joblib"))

        self._model = joblib.load(path_model)
        self._model.model_ = keras.models.load_model("model", custom_objects={'sampling': self._model.sampling})

    def set_hyperparameter(self, hyperparameter):
        """
        Sets the model hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = vae.VAE
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Creates the VAE model with the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = vae.VAE(**self._hyperparameter)