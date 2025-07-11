from pyod.models import so_gaal # so_gaal.SO_GAAL
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from .. import algorithmbase
from .. import utils


class SO_GAAL(algorithmbase.AlgorithmBase):
    """
    SO_GAAL(hyperparameter: dict)
    Single-Objective Generative Adversarial Active Learning (SO-GAAL) for unsupervised anomaly detection.
    This class implements the SO-GAAL algorithm, a generative adversarial approach for detecting anomalies in data. 
    It inherits from `algorithmbase.AlgorithmBase` and provides methods for model training, prediction, and anomaly scoring.
    The SO-GAAL model uses a generator to synthesize potential outliers and a discriminator to distinguish between real and generated samples, enabling effective anomaly detection.
    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model and training hyperparameters. See `get_default_hyperparameters` for defaults.
    Attributes
    ----------
    _model : object
        The underlying SO-GAAL model instance.
    _hyperparameter : dict
        Dictionary of hyperparameters for the model.
    Methods
    -------
    decision_function(x)
        Computes the anomaly score for the input data `x` using the trained SO-GAAL model.
    fit(x_train: np.ndarray, y_train: np.ndarray = None)
        Trains the SO-GAAL model on the provided training data.
    predict(x: np.ndarray = None) -> np.ndarray
        Returns the anomaly score for the input data `x`.
    save_model(path_model)
        Saves the trained SO-GAAL model to the specified path.
    load_model(path_model)
        Loads a SO-GAAL model from the specified path.
    set_hyperparameter(hyperparameter)
        Sets the hyperparameters for the SO-GAAL model.
    _create_model()
        Internal method to build the SO-GAAL model based on the current hyperparameters.
    Notes
    -----
    - The model is designed for unsupervised anomaly detection and does not use `y_train` during training.
    - The anomaly score is typically based on the output of the discriminator network.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the SO_GAAL model with the given hyperparameters.

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
        Trains the alad model with the training data.

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
        logging.info("Fitting SO_GAAL model...")
        logging.info("Don't use y_train in SO_GAAL model fitting, it is not used.")
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
        self._model = so_gaal.SO_GAAL
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Creates the SO_GAAL model with the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = so_gaal.SO_GAAL(**self._hyperparameter)