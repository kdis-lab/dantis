from pyod.models import qmcd # qmcd.QMCD
import numpy as np
import logging

from .. import utils
from .. import algorithmbase

logging.basicConfig(level=logging.INFO)

class QMCD(algorithmbase.AlgorithmBase):
    """
    QMCD anomaly detection model wrapper.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing hyperparameters for the QMCD algorithm.

    Methods
    -------
    fit(x_train, y_train=None)
        Fit the QMCD model using training data.
    decision_function(x)
        Compute anomaly scores for input data.
    predict(x)
        Predict outlier probabilities.
    save_model(path_model)
        Save the model to a file.
    load_model(path_model)
        Load the model from a file.
    set_hyperparameter(hyperparameter)
        Validate and set model hyperparameters.
    _create_model()
        Instantiate the QMCD model.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the QMCD detector with given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters for the QMCD model.
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
        Fit the QMCD model using training data.

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
        logging.info("Fitting QMCD model...")
        logging.info("y_train is not used in QMCD fitting.")

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
        Save the QMCD model to the specified file.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load the QMCD model from the specified file.

        Parameters
        ----------
        path_model : str
            File path to load the model from.
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
        self._model = qmcd.QMCD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the QMCD model with the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = qmcd.QMCD(**self._hyperparameter)