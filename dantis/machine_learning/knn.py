from pyod.models import knn # knn.KNN
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class KNN(algorithmbase.AlgorithmBase):
    """
    KNN(hyperparameter: dict)

    K-Nearest Neighbors (KNN) model for unsupervised anomaly detection.

    This class wraps the PyOD implementation of the KNN algorithm,
    which uses distance-based metrics to identify anomalies.

    Inherits from
    --------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the configuration for the KNN model.

    Attributes
    ----------
    _model : knn.KNN
        Instance of the PyOD KNN model.
    _hyperparameter : dict
        Dictionary storing current model hyperparameters.

    Methods
    -------
    decision_function(x)
        Returns raw anomaly scores for input data.
    fit(x_train, y_train=None)
        Trains the KNN model using the provided training data.
    predict(x)
        Returns binary anomaly predictions for input data.
    save_model(path_model)
        Saves the trained KNN model to disk.
    load_model(path_model)
        Loads the KNN model from disk.
    set_hyperparameter(hyperparameter)
        Sets and validates the hyperparameters of the model.
    _create_model()
        Instantiates the model using the provided hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the KNN model with the given hyperparameters.

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
        Fit the KNN model on training data.

        Parameters
        ----------
        x_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray, optional
            Ground truth labels (ignored in KNN).

        Returns
        -------
        np.ndarray
            Anomaly scores for the training data.

        Notes
        -----
        - This is an unsupervised model; `y_train` is not used.
        """
        logging.info("Fitting KNN model...")
        logging.info("Note: y_train is not used in KNN model fitting.")
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
        Save the trained KNN model to a file.

        Parameters
        ----------
        path_model : str
            Path to the file where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load a trained KNN model from a file.

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
            Dictionary of hyperparameters for the KNN model.
        """
        self._hyperparameter = hyperparameter
        self._model = knn.KNN  # Class reference for inspection
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the KNN model using current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = knn.KNN(**self._hyperparameter)