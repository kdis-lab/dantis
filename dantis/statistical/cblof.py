from pyod.models import cblof # cblof.CBLOF
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class CBLOF(algorithmbase.AlgorithmBase):
    """
    CBLOF (Cluster-Based Local Outlier Factor) anomaly detection model.

    This class wraps the PyOD CBLOF model for detecting outliers
    based on clustering and local outlier factor scoring.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the CBLOF algorithm.

    Methods
    -------
    fit(x_train, y_train=None)
        Fits the CBLOF model using the training data.
    decision_function(x)
        Returns the anomaly scores of the samples.
    predict(x)
        Predicts whether samples are outliers based on anomaly scores.
    save_model(path_model)
        Saves the trained model to disk.
    load_model(path_model)
        Loads the model from disk.
    set_hyperparameter(hyperparameter)
        Validates and sets hyperparameters.
    _create_model()
        Instantiates the CBLOF model with current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the CBLOF model.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the CBLOF algorithm.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x):
        """
        Compute the anomaly scores for given samples.

        Parameters
        ----------
        x : np.ndarray
            Samples to compute anomaly scores for.

        Returns
        -------
        np.ndarray
            Anomaly scores for the samples.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the CBLOF model using the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training samples.
        y_train : np.ndarray, optional
            Not used in CBLOF but present for interface compatibility.

        Returns
        -------
        np.ndarray
            Outlier scores of the training samples in [0, 1].
        """
        logging.info("Fitting CBLOF model...")
        logging.info("Note: y_train is not used in CBLOF fitting.")
        self._model.fit(x_train)

        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict whether samples are outliers based on anomaly scores.

        Parameters
        ----------
        x : np.ndarray
            Samples to predict.

        Returns
        -------
        np.ndarray
            Predicted outlier probabilities or scores.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the CBLOF model to a file.

        Parameters
        ----------
        path_model : str
            Path where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the CBLOF model from a file.

        Parameters
        ----------
        path_model : str
            Path where the model is stored.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Validate and set the hyperparameters for the CBLOF model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = cblof.CBLOF
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the CBLOF model with the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = cblof.CBLOF(**self._hyperparameter)