from statsmodels.tsa.arima.model import ARIMA as arimaModel
import numpy as np
import inspect

from .. import utils
from .. import algorithmbase

class ARIMA(algorithmbase.AlgorithmBase):
    """
    ARIMA model for anomaly detection based on squared prediction errors.

    This class wraps the statsmodels ARIMA model to detect anomalies
    by computing the squared error between the forecast and actual values.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the ARIMA model.
        Expected keys include ARIMA parameters such as 'order' and 'endog',
        and optionally 'threshold' for anomaly detection.

    Attributes
    ----------
    results : statsmodels.tsa.arima.model.ARIMAResults
        Fitted ARIMA model results after training.
    threshold : float or None
        Threshold for binary anomaly classification based on scores.

    Methods
    -------
    fit(x_train, y_train=None)
        Fits the ARIMA model on the training time series data.
    decision_function(x)
        Returns the squared error of the forecast as anomaly score.
    predict(x)
        Predicts binary anomaly labels based on a threshold.
    save_model(path_model)
        Saves the trained model to disk.
    load_model(path_model)
        Loads a trained model from disk.
    set_hyperparameter(hyperparameter)
        Validates and sets hyperparameters.
    _create_model()
        Instantiates the ARIMA model with current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the ARIMA anomaly detection model.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for ARIMA including optional 'threshold'.
        """
        super().__init__(hyperparameter=hyperparameter)
        self.results = None
        self._create_model()
        self.threshold = hyperparameter.get("threshold", None)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        Fit the ARIMA model on the training time series.

        Parameters
        ----------
        x_train : np.ndarray
            Training time series data.
        y_train : np.ndarray, optional
            Not used but present for interface consistency.

        Returns
        -------
        np.ndarray
            Anomaly scores based on squared forecast errors.
        """
        if "endog" not in self._hyperparameter or self._hyperparameter["endog"] is None:
            self._hyperparameter["endog"] = x_train

        self.results = self._model.fit()
        return self.get_anomaly_score(x_train)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the squared error between forecast and actual values as anomaly scores.

        Parameters
        ----------
        x : np.ndarray
            Input time series data.

        Returns
        -------
        np.ndarray
            Squared errors representing anomaly scores.

        Raises
        ------
        Exception
            If model is not trained prior to calling this method.
        """
        if self.results is None:
            raise Exception("Model not trained. Call fit() before decision_function().")

        forecast = self.results.forecast(steps=len(x))
        x = x.flatten()
        forecast = forecast.flatten()
        squared_errors = (x - forecast) ** 2
        return squared_errors

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict binary anomaly labels based on thresholding the anomaly scores.

        Parameters
        ----------
        x : np.ndarray
            Input time series data.

        Returns
        -------
        np.ndarray
            Binary array where 1 indicates anomaly and 0 normal.

        Notes
        -----
        If threshold is not set, it is calculated as mean + std deviation of scores.
        """
        scores = self.decision_function(x)
        if self.threshold is None:
            self.threshold = np.mean(scores) + np.std(scores)
        return (scores >= self.threshold).astype(int)

    def save_model(self, path_model: str):
        """
        Save the ARIMA model to disk.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load the ARIMA model from disk.

        Parameters
        ----------
        path_model : str
            File path to load the model from.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Validate and set the hyperparameters for the ARIMA model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters for ARIMA.
        """
        self._hyperparameter = hyperparameter
        self._model = arimaModel
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the ARIMA model using the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        endog = self._hyperparameter.get("endog")

        if (
            endog is None or
            isinstance(endog, type) or
            endog is inspect._empty or
            (hasattr(endog, '__len__') and len(endog) == 0)
        ):
            self._hyperparameter["endog"] = [1, 2]  # Default value to prevent errors
        self._model = arimaModel(**self._hyperparameter)