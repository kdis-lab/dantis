from pyod.models import ocsvm # ocsvm.OCSVM
import numpy as np
import logging

from .. import utils
from .. import algorithmbase

logging.basicConfig(level=logging.INFO)

class OCSVM(algorithmbase.AlgorithmBase):
    """
    OCSVM(hyperparameter: dict)

    One-Class Support Vector Machine (OCSVM) for anomaly detection.
    This class wraps the PyOD OCSVM implementation with added hyperparameter management
    and consistent interface.

    Inherits from
    -------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing hyperparameters for the OCSVM algorithm.

    Attributes
    ----------
    _model : ocsvm.OCSVM
        Instance of the PyOD OCSVM detector.
    _hyperparameter : dict
        Stored hyperparameters.
    
    Methods
    -------
    fit(x_train, y_train=None)
        Fit the OCSVM model on training data.
    decision_function(x)
        Compute the raw anomaly scores.
    predict(x)
        Predict outlier probabilities.
    save_model(path_model)
        Save the model to disk.
    load_model(path_model)
        Load the model from disk.
    set_hyperparameter(hyperparameter)
        Set and validate hyperparameters.
    _create_model()
        Instantiate the OCSVM model with hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the OCSVM with the given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the OCSVM model.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray):
        """
        Compute the anomaly scores for the input samples.

        Parameters
        ----------
        x : np.ndarray
            Samples to score.

        Returns
        -------
        np.ndarray
            Anomaly scores (the higher, the more abnormal).
        """
        return self._model.decision_function(x)
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        Fit the OCSVM detector on the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training samples.
        y_train : np.ndarray, optional
            Ignored for OCSVM since it's unsupervised.

        Returns
        -------
        np.ndarray
            Anomaly scores for the training data.
        """
        logging.info("Fitting OCSVM model...")
        logging.info("Don't use y_train in OCSVM model fitting, it is not used.")
        hyperparameter = self.get_hyperparameter()
        sample_weight = hyperparameter.get("sample_weight", None)
        self._model.fit(x_train, sample_weight=sample_weight)

        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict whether samples are outliers based on learned model.

        Parameters
        ----------
        x : np.ndarray
            Samples to predict.

        Returns
        -------
        np.ndarray
            Outlier scores or probabilities.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the OCSVM model to a file.

        Parameters
        ----------
        path_model : str
            Path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the OCSVM model from a file.

        Parameters
        ----------
        path_model : str
            Path from which to load the model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Set and validate hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameter dictionary.
        """
        self._hyperparameter = hyperparameter
        self._model = ocsvm.OCSVM
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the OCSVM model with current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = ocsvm.OCSVM(**self._hyperparameter)