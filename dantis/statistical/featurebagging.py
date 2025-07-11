from pyod.models import feature_bagging # feature_bagging.FeatureBagging
import numpy as np
import logging

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class FeatureBagging(algorithmbase.AlgorithmBase):
    """
    Feature Bagging anomaly detection model.

    This class wraps the PyOD FeatureBagging algorithm, which performs
    anomaly detection by bagging features.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary of hyperparameters for the Feature Bagging algorithm.

    Methods
    -------
    fit(x_train, y_train=None)
        Fit the Feature Bagging model on training data.
    decision_function(x)
        Compute anomaly scores for input data.
    predict(x)
        Predict outlier probabilities for input samples.
    save_model(path_model)
        Save the trained model to a file.
    load_model(path_model)
        Load the model from a file.
    set_hyperparameter(hyperparameter)
        Validate and set hyperparameters.
    _create_model()
        Instantiate the Feature Bagging model.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the Feature Bagging model.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the Feature Bagging algorithm.
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
            Anomaly scores for each input sample.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the Feature Bagging model on the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training samples.
        y_train : np.ndarray, optional
            Not used but accepted for interface consistency.

        Returns
        -------
        np.ndarray
            Outlier scores for the training data.
        """
        logging.info("Fitting Feature Bagging model...")
        logging.info("Note: y_train is not used in Feature Bagging fitting.")
        if x_train.shape[1] < 2:
            raise ValueError(f"Feature Bagging requires at least 2 features. Provided: {x_train.shape[1]}.")
        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict outlier probabilities for given samples.

        Parameters
        ----------
        x : np.ndarray
            Samples to predict.

        Returns
        -------
        np.ndarray
            Probability of being outlier for each sample.
        """
        return self._model.predict_proba(x)

    def save_model(self, path_model):
        """
        Save the Feature Bagging model to a file.

        Parameters
        ----------
        path_model : str
            File path where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the Feature Bagging model from a file.

        Parameters
        ----------
        path_model : str
            File path where the model is located.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Validate and set hyperparameters for the model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = feature_bagging.FeatureBagging
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the Feature Bagging model with current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = feature_bagging.FeatureBagging(**self._hyperparameter)