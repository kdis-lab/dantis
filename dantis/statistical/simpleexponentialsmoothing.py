from statsmodels.tsa.exponential_smoothing import ets 
import numpy as np
import inspect

from .. import utils
from .. import algorithmbase

class SimpleExponentialSmoothing(algorithmbase.AlgorithmBase):
    """
    Simple Exponential Smoothing model wrapper for anomaly detection.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing hyperparameters for the ETSModel.
        Optionally includes 'threshold' for anomaly detection.

    Attributes
    ----------
    results : ETSResults or None
        The fitted ETSModel results after training.
    threshold : float or None
        Threshold to decide anomalies based on squared errors.

    Methods
    -------
    __init__(hyperparameter)
        Initialize the Simple Exponential Smoothing detector.
    fit(x_train, y_train=None)
        Fit the model using the training data.
    decision_function(x)
        Calculate squared prediction errors as anomaly scores.
    predict(x)
        Predict anomaly labels based on thresholded squared errors.
    save_model(path_model)
        Save the model to a file.
    load_model(path_model)
        Load the model from a file.
    set_hyperparameter(hyperparameter)
        Validate and set the hyperparameters.
    _create_model()
        Create the ETSModel with the provided hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the Simple Exponential Smoothing detector.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for ETSModel.
        """
        super().__init__(hyperparameter=hyperparameter)
        self.results = None
        self.threshold = hyperparameter.get("threshold", None)
        self._create_model()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the model using the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training data.
        y_train : np.ndarray, optional
            Ignored.

        Returns
        -------
        np.ndarray
            Anomaly scores on training data.
        """
        if "endog" not in self._hyperparameter or self._hyperparameter["endog"] is None:
            self._hyperparameter["endog"] = x_train

        self.results = self._model.fit()
        return self.get_anomaly_score(x_train)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Calculate squared prediction errors as anomaly scores.

        Parameters
        ----------
        x : np.ndarray
            Data to score.

        Returns
        -------
        np.ndarray
            Squared errors.
        
        Raises
        ------
        Exception
            If model has not been fitted.
        """
        if self.results is None:
            raise Exception("Model not trained. Call fit() before decision_function().")

        forecast = self.results.forecast(steps=len(x)).flatten()
        x = x.flatten()
        squared_errors = (x - forecast) ** 2
        return squared_errors

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels based on thresholded squared errors.

        Parameters
        ----------
        x : np.ndarray
            Data to classify.

        Returns
        -------
        np.ndarray
            Binary anomaly labels (1 = anomaly, 0 = normal).
        """
        scores = self.decision_function(x)
        if self.threshold is None:
            self.threshold = np.mean(scores) + np.std(scores)
        return (scores >= self.threshold).astype(int)

    def save_model(self, path_model: str):
        """
        Save the model to a file.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load the model from a file.

        Parameters
        ----------
        path_model : str
            File path to load the model from.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Validate and set the hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for ETSModel.
        """
        self._hyperparameter = hyperparameter
        self._model = ets.ETSModel
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Create the ETSModel with the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        endog = self._hyperparameter.get("endog")

        if (
            endog is None or
            isinstance(endog, type) or
            endog is inspect._empty or
            (hasattr(endog, '__len__') and len(endog) == 0)
        ):
            self._hyperparameter["endog"] = [1, 2]  # Default value to allow model creation

        self._model = ets.ETSModel(**self._hyperparameter)