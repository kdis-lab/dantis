from pyod.models import sos # sos.SOS
import numpy as np
import logging

from .. import utils
from .. import algorithmbase

logging.basicConfig(level=logging.INFO)

class SOS(algorithmbase.AlgorithmBase):
    """
    SOS (Stochastic Outlier Selection) model wrapper.

    Parameters
    ----------
    hyperparameter : dict
        Hyperparameters for the SOS algorithm.

    Attributes
    ----------
    _model : sos.SOS
        The internal SOS model instance.
    _hyperparameter : dict
        Current hyperparameters for the model.

    Methods
    -------
    __init__(hyperparameter)
        Initialize the SOS detector.
    decision_function(x)
        Compute the outlier scores for input data.
    fit(x_train, y_train=None)
        Fit the SOS model using training data.
    predict(x=None)
        Predict outlier scores on new data.
    save_model(path_model)
        Save the model to a file.
    load_model(path_model)
        Load the model from a file.
    set_hyperparameter(hyperparameter)
        Set and validate hyperparameters.
    _create_model()
        Create the SOS model instance using current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the SOS detector.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary containing hyperparameters for SOS.
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
        Fit the SOS model.

        Parameters
        ----------
        x_train : np.ndarray
            Training data.
        y_train : np.ndarray, optional
            Not used in SOS.

        Returns
        -------
        np.ndarray
            Outlier scores on the training data.
        """
        logging.info("Fitting SOS model...")
        logging.info("Don't use y_train in SOS model fitting, it is not used.")
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
        Save the SOS model.

        Parameters
        ----------
        path_model : str
            Path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load the SOS model.

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
        self._model = sos.SOS
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Create the SOS model instance with current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = sos.SOS(**self._hyperparameter)