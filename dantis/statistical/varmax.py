from statsmodels.tsa.statespace import varmax
import numpy as np
import importlib

from .. import utils
from .. import algorithmbase

class VARMAX(algorithmbase.AlgorithmBase):
    """
    VARMAX model wrapper for multivariate time series anomaly detection using prediction errors.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing VARMAX model hyperparameters.
        Optional 'threshold' key for anomaly decision boundary.

    Attributes
    ----------
    results : varmax.VARMAXResults or None
        The fitted model results after training.
    _model : varmax.VARMAX or None
        The internal VARMAX model instance (created during fit).
    threshold : float or None
        Threshold used to classify anomalies based on prediction error scores.

    Methods
    -------
    __init__(hyperparameter)
        Initialize the VARMAX detector with given hyperparameters.
    decision_function(x)
        Compute anomaly scores as mean squared prediction errors.
    fit(x_train, y_train=None)
        Fit the VARMAX model on training data.
    predict(x)
        Predict binary anomaly labels using computed scores and threshold.
    save_model(path_model)
        Save the model to a file.
    load_model(path_model)
        Load the model from a file.
    set_hyperparameter(hyperparameter)
        Validate and set model hyperparameters.
    _create_model(x=None, y=None)
        Placeholder for model creation; actual model created during fit.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the VARMAX detector.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for VARMAX.
        """
        super().__init__(hyperparameter=hyperparameter)
        self.results = None
        self._model = None  # Model will be created during fit
        self.threshold = hyperparameter.get("threshold", None)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores as mean squared prediction errors.

        Parameters
        ----------
        x : np.ndarray
            Data to evaluate.

        Returns
        -------
        np.ndarray
            Mean squared errors per sample as anomaly scores.

        Raises
        ------
        Exception
            If the model has not been trained yet.
        """
        if self.results is None:
            raise Exception("Model not trained. Call fit() before decision_function().")

        forecast = self.results.forecast(steps=len(x))
        x = np.asarray(x, dtype=float)
        forecast = np.asarray(forecast)

        squared_errors = (x - forecast) ** 2
        return np.mean(squared_errors, axis=1)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        Fit the VARMAX model on training data.

        Parameters
        ----------
        x_train : np.ndarray
            Multivariate training data.
        y_train : np.ndarray, optional
            Ignored.

        Returns
        -------
        np.ndarray
            Anomaly scores on the training data.
        """
        x_train = np.asarray(x_train, dtype=float)
        if x_train.ndim == 1:
            x_train = x_train[:, np.newaxis]

        self._hyperparameter["endog"] = x_train

        # Ensure exogenous variables are also floats if provided
        if "exog" in self._hyperparameter and self._hyperparameter["exog"] is not None:
            self._hyperparameter["exog"] = np.asarray(self._hyperparameter["exog"], dtype=float)

        self._model = varmax.VARMAX(**self._hyperparameter)
        self.results = self._model.fit(disp=False)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict binary anomaly labels based on thresholded anomaly scores.

        Parameters
        ----------
        x : np.ndarray
            Data for prediction.

        Returns
        -------
        np.ndarray
            Binary anomaly labels: 1 for anomaly, 0 for normal.
        """
        scores = self.decision_function(x)
        if self.threshold is None:
            self.threshold = np.mean(scores) + np.std(scores)
        return (scores >= self.threshold).astype(int)

    def save_model(self, path_model):
        """
        Save the VARMAX model to a file.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the VARMAX model from a file.

        Parameters
        ----------
        path_model : str
            File path to load the model from.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Set and validate hyperparameters for the VARMAX model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of model hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = varmax.VARMAX  # reference for inspection
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

        # Minimal conversion if 'endog' already defined
        if self.check_hyperparams and "endog" in self._hyperparameter:
            endog = np.asarray(self._hyperparameter["endog"])
            if endog.ndim == 1:
                self._hyperparameter["endog"] = endog[:, np.newaxis]

    def _create_model(self, x: np.array = None, y: np.array = None):
        """
        Placeholder method. Model creation is deferred until fit.

        Parameters
        ----------
        x : np.array, optional
            Ignored.
        y : np.array, optional
            Ignored.
        """
        self.set_hyperparameter(self._hyperparameter)