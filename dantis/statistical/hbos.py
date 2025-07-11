from pyod.models import hbos # hbos.HBOS
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class HBOS(algorithmbase.AlgorithmBase):
    """
    Histogram-based Outlier Score (HBOS) detector.

    This class wraps the HBOS model from PyOD, which detects anomalies
    based on histograms of feature distributions.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary of hyperparameters for the HBOS algorithm.

    Methods
    -------
    fit(x_train, y_train=None)
        Fits the HBOS model on training data.
    decision_function(x)
        Computes anomaly scores for the given samples.
    predict(x)
        Predicts outlier probabilities for given samples.
    save_model(path_model)
        Saves the trained model to a file.
    load_model(path_model)
        Loads the model from a file.
    set_hyperparameter(hyperparameter)
        Validates and sets the hyperparameters.
    _create_model()
        Instantiates the HBOS model using hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the HBOS detector.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the HBOS algorithm.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x):
        """
        Compute the anomaly scores for input samples.

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
        Fit the HBOS model on training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training data.
        y_train : np.ndarray, optional
            Not used, present for compatibility.

        Returns
        -------
        np.ndarray
            Outlier scores for training data.
        """
        logging.info("Fitting HBOS model...")
        logging.info("Note: y_train is not used in HBOS fitting.")
        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict the probability of samples being outliers.

        Parameters
        ----------
        x : np.ndarray
            Samples to predict.

        Returns
        -------
        np.ndarray
            Outlier probabilities.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the HBOS model to a file.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the HBOS model from a file.

        Parameters
        ----------
        path_model : str
            File path to load the model from.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Set and validate hyperparameters for the HBOS model.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters dictionary.
        """
        self._hyperparameter = hyperparameter
        self._model = hbos.HBOS
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the HBOS model with the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = hbos.HBOS(**self._hyperparameter)