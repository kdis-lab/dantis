from pyod.models import kpca # kpca.KPCA
import numpy as np
import logging

from .. import utils
from .. import algorithmbase

logging.basicConfig(level=logging.INFO)

class KPCA(algorithmbase.AlgorithmBase):
    """
    KPCA anomaly detection model wrapper.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the KPCA model.

    Methods
    -------
    fit(x_train, y_train=None)
        Fits the KPCA model using training data.
    decision_function(x)
        Computes anomaly scores for given data.
    predict(x)
        Predicts outliers based on the model.
    save_model(path_model)
        Saves the model to a file.
    load_model(path_model)
        Loads the model from a file.
    set_hyperparameter(hyperparameter)
        Validates and sets hyperparameters.
    _create_model()
        Creates the KPCA model instance.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize KPCA detector with given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the KPCA algorithm.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray):
        """
        Compute anomaly scores for input data.

        Parameters
        ----------
        x : np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Anomaly scores.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the KPCA model using training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training data.
        y_train : np.ndarray, optional
            Not used.

        Returns
        -------
        np.ndarray
            Outlier scores of the training data [0,1].
        """
        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict whether samples are outliers.

        Parameters
        ----------
        x : np.ndarray, optional
            Samples to predict.

        Returns
        -------
        np.ndarray
            Probability of being outlier.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the KPCA model to a file.

        Parameters
        ----------
        path_model : str
            Path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the KPCA model from a file.

        Parameters
        ----------
        path_model : str
            Path to load the model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Validate and set hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary with hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = kpca.KPCA
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the KPCA model with hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = kpca.KPCA(**self._hyperparameter)