from pyod.models import gmm # gmm.GMM
import numpy as np
import logging

from .. import utils
from .. import algorithmbase

logging.basicConfig(level=logging.INFO)

class GMM(algorithmbase.AlgorithmBase):
    """
    Gaussian Mixture Model (GMM) for anomaly detection.

    This class wraps the PyOD GMM model which fits a Gaussian Mixture Model
    to the data and uses it to detect outliers based on their likelihood.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the GMM model.

    Methods
    -------
    fit(x_train, y_train=None)
        Fits the GMM model on the training data.
    decision_function(x)
        Computes the anomaly score for given samples.
    predict(x)
        Predicts the outlier probability for given samples.
    save_model(path_model)
        Saves the trained model to a file.
    load_model(path_model)
        Loads the model from a file.
    set_hyperparameter(hyperparameter)
        Validates and sets the model hyperparameters.
    _create_model()
        Instantiates the GMM model using the hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the GMM anomaly detector.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the GMM algorithm.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x):
        """
        Compute the decision function (anomaly scores) for the input data.

        Parameters
        ----------
        x : np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Anomaly scores for the input samples.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        Fit the GMM model on the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training samples.
        y_train : np.ndarray, optional
            Not used but accepted for compatibility.

        Returns
        -------
        np.ndarray
            Outlier scores for the training data.
        """
        logging.info("Fitting GMM model...")
        logging.info("Note: y_train is not used in GMM model fitting.")
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
            Probability of being an outlier for each sample.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the GMM model to a file.

        Parameters
        ----------
        path_model : str
            File path where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the GMM model from a file.

        Parameters
        ----------
        path_model : str
            File path where the model is located.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Validate and set the hyperparameters for the model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = gmm.GMM
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the GMM model using the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = gmm.GMM(**self._hyperparameter)