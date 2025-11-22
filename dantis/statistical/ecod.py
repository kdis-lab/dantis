from pyod.models import ecod # ecod.ECOD
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

# Set up logging for model information output
logging.basicConfig(level=logging.INFO)

class ECOD(algorithmbase.AlgorithmBase):
    """
    ECOD(hyperparameter: dict)

    Empirical Cumulative Distribution Function Outlier Detection (ECOD).

    This class is a wrapper around PyOD's ECOD model for unsupervised anomaly detection.
    ECOD is a non-parametric algorithm that ranks outliers using empirical distributions.
    It does not require labeled data for training and works well for high-dimensional problems.

    Inherits from:
    ---------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model hyperparameters. See `get_default_hyperparameters` for defaults.

    Attributes
    ----------
    _model : ecod.ECOD
        Instance of the ECOD anomaly detection model.
    _hyperparameter : dict
        Dictionary of hyperparameters for model instantiation.

    Methods
    -------
    decision_function(x)
        Returns raw anomaly scores for input data.
    fit(x_train, y_train=None)
        Fits the ECOD model using only the input features (unsupervised).
    predict(x)
        Predicts binary anomaly labels for input data based on the threshold.
    save_model(path_model)
        Saves the ECOD model to the given path.
    load_model(path_model)
        Loads the ECOD model from the specified path.
    set_hyperparameter(hyperparameter)
        Sets and validates the model hyperparameters.
    _create_model()
        Internal method to initialize the ECOD model.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the ECOD model with given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of model hyperparameters.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Computes raw anomaly scores for input samples.

        Parameters
        ----------
        x : np.ndarray
            Input data to score.

        Returns
        -------
        np.ndarray
            Anomaly scores, where higher values indicate greater likelihood of being an outlier.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fits the ECOD model using unsupervised training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training data (features only).
        y_train : np.ndarray, optional
            Target labels (ignored in ECOD).

        Returns
        -------
        np.ndarray
            Anomaly scores for the training data.

        Notes
        -----
        - This method ignores y_train, as ECOD is fully unsupervised.
        - Logging is used to indicate that y_train is not utilized.
        """
        logging.info("Fitting ECOD model...")
        logging.info("Note: y_train is not used in ECOD training.")
        self._model.fit(x_train)

        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predicts binary anomaly labels for input data.

        Parameters
        ----------
        x : np.ndarray
            Input data to classify.

        Returns
        -------
        np.ndarray
            Binary array where 1 indicates an anomaly and 0 indicates a normal instance.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model: str):
        """
        Saves the trained ECOD model to the specified file path.

        Parameters
        ----------
        path_model : str
            Destination path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Loads a previously saved ECOD model from disk.

        Parameters
        ----------
        path_model : str
            Path of the saved model file.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Sets and validates the model's hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary containing ECOD model parameters.
        """
        self._hyperparameter = hyperparameter
        self._model = ecod.ECOD  # Reference to class for introspection
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Internal method to instantiate the ECOD model with the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = ecod.ECOD(**self._hyperparameter)