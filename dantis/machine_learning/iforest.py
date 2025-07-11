from pyod.models import iforest # iforest.IForest
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class IForest(algorithmbase.AlgorithmBase):
    """
    IForest(hyperparameter: dict)

    Isolation Forest (IForest) for unsupervised anomaly detection.

    This class wraps the PyOD implementation of the Isolation Forest algorithm,
    which isolates anomalies instead of profiling normal data points.
    Anomalies are easier to isolate, and therefore receive higher anomaly scores.

    Inherits from
    --------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model configuration values. See `get_default_hyperparameters` for defaults.

    Attributes
    ----------
    _model : iforest.IForest
        Instance of the PyOD IForest model.
    _hyperparameter : dict
        Dictionary of model hyperparameters.

    Methods
    -------
    decision_function(x)
        Returns raw anomaly scores for input data.
    fit(x_train, y_train=None)
        Fits the IForest model using training data (unsupervised).
    predict(x)
        Returns binary anomaly predictions for input samples.
    save_model(path_model)
        Saves the trained model to a specified file path.
    load_model(path_model)
        Loads the model from a specified file.
    set_hyperparameter(hyperparameter)
        Sets and validates model hyperparameters.
    _create_model()
        Internal method to instantiate the model using current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initializes the IForest model with the provided hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary containing model parameters.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Computes raw anomaly scores for the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data for scoring.

        Returns
        -------
        np.ndarray
            Raw anomaly scores, where higher values indicate a higher likelihood of being an outlier.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fits the IForest model to the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray, optional
            Target labels (not used in IForest).

        Returns
        -------
        np.ndarray
            Anomaly scores for the training data.

        Notes
        -----
        - This is an unsupervised model; `y_train` is ignored.
        - Logs fitting process for visibility.
        """
        logging.info("Fitting IForest model...")
        logging.info("Note: y_train is not used in IForest model fitting.")
        self._model.fit(x_train)

        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predicts binary anomaly labels for the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data to classify.

        Returns
        -------
        np.ndarray
            Array of binary predictions: 1 for anomaly, 0 for normal.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model: str):
        """
        Saves the trained model to a file.

        Parameters
        ----------
        path_model : str
            Path where the model should be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Loads a trained model from a file.

        Parameters
        ----------
        path_model : str
            Path of the file containing the saved model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Sets and validates the model's hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = iforest.IForest  # Class reference for introspection
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiates the IForest model using the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = iforest.IForest(**self._hyperparameter)