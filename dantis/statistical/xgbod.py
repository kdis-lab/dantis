from pyod.models import xgbod # xgbod.XGBOD
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class XGBOD(algorithmbase.AlgorithmBase):
    """
    XGBOD (Extreme Gradient Boosting for Outlier Detection) model wrapper.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing hyperparameters for the XGBOD algorithm.

    Methods
    -------
    decision_function(x)
        Compute the outlier scores for input data.
    fit(x_train, y_train=None)
        Fit the XGBOD model on training data.
    predict(x)
        Predict outlier scores for new data.
    save_model(path_model)
        Save the trained model to the given path.
    load_model(path_model)
        Load the model from the given path.
    set_hyperparameter(hyperparameter)
        Validate and set model hyperparameters.
    _create_model()
        Instantiate the XGBOD model with current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the XGBOD detector.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters for XGBOD.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute outlier scores for input samples.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Outlier scores for the input samples.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the XGBOD model using the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training data features.
        y_train : np.ndarray, optional
            Training data labels (not mandatory for XGBOD).

        Returns
        -------
        np.ndarray
            Outlier scores on the training data.
        """
        self._model.fit(x_train, y_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict outlier scores for new data.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Outlier scores for the input samples.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model: str):
        """
        Save the trained XGBOD model to a file.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load a trained XGBOD model from a file.

        Parameters
        ----------
        path_model : str
            File path from which to load the model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Validate and set hyperparameters for the XGBOD model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = xgbod.XGBOD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the XGBOD model using the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = xgbod.XGBOD(**self._hyperparameter)