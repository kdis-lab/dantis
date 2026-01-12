from pyod.models import abod # abod.ABOD
import numpy as np

from .. import algorithmbase
from .. import utils

class ABOD(algorithmbase.AlgorithmBase):
    """
    Angle-Based Outlier Detection (ABOD) model for unsupervised anomaly detection.

    This class wraps the ABOD model from the pyod library,
    providing consistent hyperparameter management and model interface.

    Inherits from
    -------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the ABOD model.

    Attributes
    ----------
    _model : pyod.models.abod.ABOD
        The underlying ABOD model instance.
    _hyperparameter : dict
        Stored hyperparameters for the model.

    Methods
    -------
    fit(x_train, y_train=None)
        Fit the ABOD model on training data.
    decision_function(x)
        Compute the outlier scores of given samples.
    predict(x)
        Predict anomaly scores for test samples.
    save_model(path_model)
        Save the trained model to a file.
    load_model(path_model)
        Load a trained model from a file.
    set_hyperparameter(hyperparameter)
        Validate and set hyperparameters.
    _create_model()
        Instantiate the ABOD model with the current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the ABOD model with specified hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the ABOD model.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x):
        """
        Compute the anomaly scores for the input samples.

        Parameters
        ----------
        x : np.ndarray
            Input samples to score.

        Returns
        -------
        np.ndarray
            Outlier scores.
        """
        return self._model.decision_function(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the ABOD model on the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray, optional
            Not used by ABOD but kept for interface consistency.

        Returns
        -------
        np.ndarray
            Outlier scores for the training data.
        """
        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict anomaly scores for given samples.

        Parameters
        ----------
        x : np.ndarray, optional
            Samples to predict.

        Returns
        -------
        np.ndarray
            Predicted anomaly scores.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the trained ABOD model to disk.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load a trained ABOD model from disk.

        Parameters
        ----------
        path_model : str
            File path from which to load the model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Validate and set the hyperparameters for the ABOD model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = abod.ABOD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the ABOD model with the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = abod.ABOD(**self._hyperparameter)