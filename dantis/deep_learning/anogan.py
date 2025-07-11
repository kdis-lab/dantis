from pyod.models import anogan # anogan.AnoGAN
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from .. import algorithmbase
from .. import utils


class AnoGAN(algorithmbase.AlgorithmBase):
    """
    AnoGAN(hyperparameter: dict)
    Generative Adversarial Network for anomaly detection (AnoGAN).
    This class implements the AnoGAN architecture for unsupervised anomaly detection, leveraging a generative adversarial network to learn the distribution of normal data and identify anomalies based on reconstruction and discriminator loss. It inherits from `algorithmbase.AlgorithmBase` and provides methods for model training, prediction, and anomaly scoring.
    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model and training hyperparameters. See `get_default_hyperparameters` for defaults.
    Attributes
    ----------
    _model : object
        The underlying AnoGAN model instance.
    _hyperparameter : dict
        Dictionary of hyperparameters for the model.
    Methods
    -------
    decision_function(x)
        Computes the anomaly score for the input data `x` using the trained AnoGAN model.
    fit(x_train: np.array, y_train: np.array = None)
        Trains the AnoGAN model on the provided training data.
    predict(x: np.array = None)
        Returns the anomaly score for the input data `x`.
    save_model(path_model)
        Saves the trained model to the specified path.
    load_model(path_model)
        Loads a model from the specified path.
    set_hyperparameter(hyperparameter)
        Sets the hyperparameters for the model.
    _create_model()
        Internal method to build the AnoGAN architecture based on the current hyperparameters.
    Notes
    -----
    - The model is designed for unsupervised anomaly detection and does not use labels during training.
    - The anomaly score is typically based on a combination of reconstruction error and discriminator loss.
    - Input data should be preprocessed according to the requirements of the underlying AnoGAN implementation.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the AnoGAN model with the given hyperparameters.

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
        Trains the AnoGAN model with the training data.

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
        logging.info("Fitting AnoGAN model...")
        logging.info("Don't use y_train in AnoGAN model fitting, it is not used.")
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
        self._model = anogan.AnoGAN
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Internal method to build the anogan model architecture.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = anogan.AnoGAN(**self._hyperparameter)