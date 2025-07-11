from pyod.models import loda # loda.LODA
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class LODA(algorithmbase.AlgorithmBase):
    """
    LODA anomaly detection model wrapper.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the LODA algorithm.

    Methods
    -------
    fit(x_train, y_train=None)
        Fit the LODA model on training data.
    decision_function(x)
        Compute anomaly scores for input data.
    predict(x)
        Predict outliers for given samples.
    save_model(path_model)
        Save model to disk.
    load_model(path_model)
        Load model from disk.
    set_hyperparameter(hyperparameter)
        Validate and set hyperparameters.
    _create_model()
        Instantiate the LODA model.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the LODA detector with given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the LODA algorithm.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray):
        """
        Compute the decision function (anomaly scores) for the given input data.

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
        Fit the LODA model using the training data.

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
            Outlier prediction probabilities.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the LODA model to a file.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the LODA model from a file.

        Parameters
        ----------
        path_model : str
            File path to load the model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Validate and set the hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters dictionary.
        """
        self._hyperparameter = hyperparameter
        self._model = loda.LODA
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the LODA model using the set hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = loda.LODA(**self._hyperparameter)