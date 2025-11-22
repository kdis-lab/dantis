from pyod.models import sod # sod.SOD
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class SOD(algorithmbase.AlgorithmBase):
    """
    SOD (Subspace Outlier Detection) model wrapper.

    Parameters
    ----------
    hyperparameter : dict
        Hyperparameters for the SOD algorithm.

    Attributes
    ----------
    _model : sod.SOD
        The internal SOD model instance.
    _hyperparameter : dict
        Current hyperparameters for the model.

    Methods
    -------
    __init__(hyperparameter)
        Initialize the SOD detector.
    decision_function(x)
        Compute the outlier scores for input data.
    fit(x_train, y_train=None)
        Fit the SOD model using training data.
    predict(x=None)
        Predict outlier scores on new data.
    save_model(path_model)
        Save the model to a file.
    load_model(path_model)
        Load the model from a file.
    set_hyperparameter(hyperparameter)
        Set and validate hyperparameters.
    _create_model()
        Create the SOD model instance using current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the SOD detector.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary containing hyperparameters for SOD.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the decision function (outlier scores).

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Outlier scores.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the SOD model.

        Parameters
        ----------
        x_train : np.ndarray
            Training data.
        y_train : np.ndarray, optional
            Not used in SOD.

        Returns
        -------
        np.ndarray
            Outlier scores on the training data.
        """
        logging.info("Fitting SOD model...")
        logging.info("Don't use y_train in SOD model fitting, it is not used.")
        self._model.fit(x_train)
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
            Outlier scores (probabilities).
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model: str):
        """
        Save the SOD model.

        Parameters
        ----------
        path_model : str
            Path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load the SOD model.

        Parameters
        ----------
        path_model : str
            Path to load the model from.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Set and validate hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = sod.SOD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Create the SOD model instance with current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = sod.SOD(**self._hyperparameter)