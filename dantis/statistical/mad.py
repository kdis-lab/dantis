from pyod.models import mad # mad.MAD
import numpy as np
import logging

from .. import utils
from .. import algorithmbase

logging.basicConfig(level=logging.INFO)

class MAD(algorithmbase.AlgorithmBase):
    """
    MAD anomaly detection model wrapper.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the MAD algorithm.

    Methods
    -------
    fit(x_train, y_train=None)
        Fit the MAD model using training data.
    decision_function(x)
        Compute anomaly scores for input data.
    predict(x)
        Predict outlier probabilities.
    save_model(path_model)
        Save model to disk.
    load_model(path_model)
        Load model from disk.
    set_hyperparameter(hyperparameter)
        Validate and set hyperparameters.
    _create_model()
        Instantiate the MAD model.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the MAD detector with given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters for the MAD model.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the decision function (anomaly scores) for the input data.

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
        Fit the MAD model using training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training data.
        y_train : np.ndarray, optional
            Not used.

        Returns
        -------
        np.ndarray
            Outlier scores of the training data in [0, 1].
        """
        logging.info("Fitting MAD model...")
        logging.info("y_train is not used in MAD fitting.")

        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict outlier probabilities for the input data.

        Parameters
        ----------
        x : np.ndarray, optional
            Samples to predict.

        Returns
        -------
        np.ndarray
            Outlier probabilities.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model: str):
        """
        Save the MAD model to the specified file.

        Parameters
        ----------
        path_model : str
            File path where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load the MAD model from the specified file.

        Parameters
        ----------
        path_model : str
            File path where the saved model is located.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Validate and set the hyperparameters of the model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary containing the model's hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = mad.MAD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the MAD model with the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = mad.MAD(**self._hyperparameter)