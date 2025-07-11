from pyod.models import copod # copod.COPOD
import numpy as np
import logging

from .. import utils
from .. import algorithmbase

logging.basicConfig(level=logging.INFO)

class COPOD(algorithmbase.AlgorithmBase):
    """
    COPOD (Copula-Based Outlier Detection) anomaly detection model.

    This class wraps the PyOD COPOD model, which detects outliers using
    copula-based methods.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the COPOD algorithm.

    Methods
    -------
    fit(x_train, y_train=None)
        Fits the COPOD model using the training data.
    decision_function(x)
        Computes anomaly scores for input samples.
    predict(x)
        Predicts outlier scores for given samples.
    save_model(path_model)
        Saves the trained model to a file.
    load_model(path_model)
        Loads the model from a file.
    set_hyperparameter(hyperparameter)
        Validates and sets the model hyperparameters.
    _create_model()
        Instantiates the COPOD model with current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the COPOD model.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the COPOD algorithm.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x):
        """
        Compute the decision function (anomaly scores) for input data.

        Parameters
        ----------
        x : np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Decision function (outlier) scores.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the COPOD model on the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training samples.
        y_train : np.ndarray, optional
            Not used but accepted for interface consistency.

        Returns
        -------
        np.ndarray
            Outlier scores in [0,1] for the training data.
        """
        logging.info("Fitting COPOD model...")
        logging.info("Note: y_train is not used in COPOD fitting.")
        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict outlier scores for given samples.

        Parameters
        ----------
        x : np.ndarray
            Samples to predict.

        Returns
        -------
        np.ndarray
            Predicted outlier scores.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the COPOD model to a file.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the COPOD model from a file.

        Parameters
        ----------
        path_model : str
            File path to load the model from.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Validate and set the hyperparameters for the COPOD model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = copod.COPOD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the COPOD model with current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = copod.COPOD(**self._hyperparameter)