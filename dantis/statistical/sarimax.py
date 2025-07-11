from statsmodels.tsa.statespace import sarimax
import numpy as np

from .. import utils
from .. import algorithmbase


class SARIMAX(algorithmbase.AlgorithmBase):
    """
    SARIMAX model wrapper for anomaly detection using prediction errors.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing SARIMAX model hyperparameters.
        Optional 'threshold' key can be provided to set anomaly detection threshold.

    Attributes
    ----------
    results : SARIMAXResultsWrapper or None
        The fitted model results after training.
    threshold : float or None
        Threshold for deciding anomalies based on squared errors.

    Methods
    -------
    __init__(hyperparameter)
        Initialize the SARIMAX detector.
    decision_function(x)
        Compute anomaly scores as squared prediction errors.
    fit(x_train, y_train=None)
        Fit the SARIMAX model on training data.
    predict(x)
        Predict anomaly labels based on squared prediction errors and threshold.
    save_model(path_model)
        Save the model to a file.
    load_model(path_model)
        Load the model from a file.
    set_hyperparameter(hyperparameter)
        Prepare and validate hyperparameters without creating the model.
    _create_model()
        Model creation deferred until fit, since data is needed.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the SARIMAX detector.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for SARIMAX.
        """
        super().__init__(hyperparameter=hyperparameter)
        self.results = None
        self._model = None  # Will be created during fit
        self.threshold = hyperparameter.get("threshold", None)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute anomaly scores as squared prediction errors.

        Parameters
        ----------
        x : np.ndarray
            Real data points to compare against the forecast.

        Returns
        -------
        np.ndarray
            Squared errors as anomaly scores.
        
        Raises
        ------
        Exception
            If model is not yet fitted.
        """
        if self.results is None:
            raise Exception("Model not trained. Call fit() before decision_function().")

        forecast = self.results.forecast(steps=len(x)).flatten()
        x = x.flatten()
        squared_errors = (x - forecast) ** 2
        return squared_errors

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the SARIMAX model on training data.

        Parameters
        ----------
        x_train : np.ndarray
            Univariate training data.
        y_train : np.ndarray, optional
            Ignored.

        Returns
        -------
        np.ndarray
            Anomaly scores for the training data.
        """
        self._hyperparameter["endog"] = x_train
        self._model = sarimax.SARIMAX(**self._hyperparameter)
        self.results = self._model.fit(disp=False)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict anomaly labels based on squared prediction errors and threshold.

        Parameters
        ----------
        x : np.ndarray
            Data to classify as anomaly or normal.

        Returns
        -------
        np.ndarray
            Binary labels: 1 for anomaly, 0 for normal.
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
            Path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load the model from a file.

        Parameters
        ----------
        path_model : str
            Path to load the model from.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Prepare and validate hyperparameters without creating the model.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for SARIMAX.
        """
        self._hyperparameter = hyperparameter
        self._model = sarimax.SARIMAX  # for inspection
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Model creation deferred until fit, since data is needed.
        """
        self.set_hyperparameter(self._hyperparameter)