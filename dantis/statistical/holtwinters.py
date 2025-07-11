
from statsmodels.tsa import holtwinters as holtWintersModel
import numpy as np
import inspect

from .. import utils
from .. import algorithmbase

class HoltWinters(algorithmbase.AlgorithmBase):
    """
    Holt-Winters forecasting model wrapper for anomaly detection.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary of hyperparameters for the Holt Winters model.
        Expected to contain the "endog" key with the time series data.

    Attributes
    ----------
    results : HoltWintersResults
        The fitted Holt Winters model results.
    threshold : float or None
        Threshold for anomaly detection. If None, computed automatically.

    Methods
    -------
    fit(x_train, y_train=None)
        Fits the Holt Winters model to the training data.
    decision_function(x)
        Computes squared error between forecast and actual values as anomaly score.
    predict(x)
        Predicts anomalies based on the squared error and threshold.
    save_model(path_model)
        Saves the model state to disk.
    load_model(path_model)
        Loads the model state from disk.
    set_hyperparameter(hyperparameter)
        Sets and validates hyperparameters.
    _create_model()
        Creates the Holt Winters model instance.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize HoltWinters anomaly detector.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the Holt Winters model.
        """
        super().__init__(hyperparameter=hyperparameter)
        self.results = None
        self.threshold = hyperparameter.get("threshold", None)
        self._create_model()

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        Fit the Holt Winters model on training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training data.
        y_train : np.ndarray, optional
            Not used but included for interface compatibility.

        Returns
        -------
        np.ndarray
            Outlier scores based on squared errors of the training data.
        """
        self.results = self._model.fit()
        return self.get_anomaly_score(x_train)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores as squared forecast errors.

        Parameters
        ----------
        x : np.ndarray
            Data to compute anomaly scores for.

        Returns
        -------
        np.ndarray
            Squared error scores.
        
        Raises
        ------
        Exception
            If called before the model is fitted.
        """
        if self.results is None:
            raise Exception("Modelo no entrenado. Llama a fit() antes de usar decision_function.")

        forecast = self.results.forecast(steps=len(x))
        x = x.flatten()
        forecast = forecast.flatten()
        squared_errors = (x - forecast) ** 2
        return squared_errors

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict whether samples are outliers based on anomaly scores.

        Parameters
        ----------
        x : np.ndarray
            Samples to predict.

        Returns
        -------
        np.ndarray
            Binary array where 1 indicates outlier and 0 indicates normal.
        """
        scores = self.decision_function(x)
        if self.threshold is None:
            self.threshold = np.mean(scores) + np.std(scores)
        return (scores >= self.threshold).astype(int)

    def save_model(self, path_model):
        """
        Save the Holt Winters model to a file.

        Parameters
        ----------
        path_model : str
            Path where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the Holt Winters model from a file.

        Parameters
        ----------
        path_model : str
            Path from where to load the model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Set and validate hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters dictionary.
        """
        self._hyperparameter = hyperparameter
        self._model = holtWintersModel.Holt
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Create Holt Winters model instance with validated hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        endog = self._hyperparameter.get("endog")

        if (
            endog is None or
            isinstance(endog, type) or
            endog is inspect._empty or
            (hasattr(endog, '__len__') and len(endog) == 0)
        ):
            self._hyperparameter["endog"] = [1, 2]  # Default value if none provided

        self._model = holtWintersModel.Holt(**self._hyperparameter)