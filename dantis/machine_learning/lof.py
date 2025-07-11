from pyod.models import lof # lof.LOF
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class LOF(algorithmbase.AlgorithmBase):
    """
    LOF(hyperparameter: dict)

    Local Outlier Factor (LOF) model for unsupervised anomaly detection.

    This class wraps the PyOD implementation of the LOF algorithm,
    which uses local density deviation to identify outliers.

    Inherits from
    --------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the configuration for the LOF model.

    Attributes
    ----------
    _model : lof.LOF
        Instance of the PyOD LOF model.
    _hyperparameter : dict
        Dictionary storing current model hyperparameters.

    Methods
    -------
    decision_function(x)
        Returns raw anomaly scores for input data.
    fit(x_train, y_train=None)
        Trains the LOF model using the provided training data.
    predict(x)
        Returns binary anomaly predictions for input data.
    save_model(path_model)
        Saves the trained LOF model to disk.
    load_model(path_model)
        Loads the LOF model from disk.
    set_hyperparameter(hyperparameter)
        Sets and validates the hyperparameters of the model.
    _create_model()
        Instantiates the model using the provided hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the LOF model with the given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary containing model hyperparameters.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute raw outlier scores for input data.

        Parameters
        ----------
        x : np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Outlier scores (higher means more anomalous).
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the LOF model on training data.

        Parameters
        ----------
        x_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray, optional
            Ground truth labels (ignored in LOF).

        Returns
        -------
        np.ndarray
            Anomaly scores for the training data.

        Notes
        -----
        - This is an unsupervised model; `y_train` is not used.
        """
        logging.info("Fitting LOF model...")
        logging.info("Note: y_train is not used in LOF model fitting.")
        self._model.fit(x_train)

        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict binary anomaly labels for input samples.

        Parameters
        ----------
        x : np.ndarray
            Input data to classify.

        Returns
        -------
        np.ndarray
            Binary predictions (1 = outlier, 0 = inlier).
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model: str):
        """
        Save the trained LOF model to a file.

        Parameters
        ----------
        path_model : str
            Path to the file where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load a trained LOF model from a file.

        Parameters
        ----------
        path_model : str
            Path to the saved model file.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Set and validate model hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters for the LOF model.
        """
        self._hyperparameter = hyperparameter
        self._model = lof.LOF  # Class reference for inspection
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the LOF model using current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = lof.LOF(**self._hyperparameter)