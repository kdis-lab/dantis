from pyod.models import lunar  # lunar.LUNAR
import numpy as np
import logging
import importlib

from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class LUNAR(algorithmbase.AlgorithmBase):
    """
    LUNAR anomaly detection model wrapper.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the LUNAR algorithm.

    Methods
    -------
    fit(x_train, y_train=None)
        Fit the LUNAR model using training data.
    decision_function(x)
        Compute anomaly scores for input data.
    predict(x)
        Predict outlier probabilities.
    save_model(path_model)
        Save model to disk.
    load_model(path_model)
        Load model from disk.
    set_hyperparameter(hyperparameter)
        Validate and set hyperparameters, including optional scaler deserialization.
    _create_model()
        Instantiate the LUNAR model.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the LUNAR detector with given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters for the LUNAR model.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the decision function (anomaly scores) for the given input data.

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
        Fit the LUNAR model using the training data.

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
        logging.info("Fitting LUNAR model...")
        logging.info("y_train is not used in LUNAR fitting.")

        # If scaler exists and has a fit method, fit it on training data
        if hasattr(self._model, "scaler") and hasattr(self._model.scaler, "fit"):
            self._model.scaler.fit(x_train)

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
        Save the LUNAR model to the specified file.

        Parameters
        ----------
        path_model : str
            File path where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load the LUNAR model from the specified file.

        Parameters
        ----------
        path_model : str
            File path where the saved model is located.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Validate and set the hyperparameters of the model.

        Handles deserialization of scaler if specified.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters for the model.
        """
        self._hyperparameter = hyperparameter
        self._model = lunar.LUNAR  # for signature inspection
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

        # Deserialize scaler if specified
        if "scaler" in self._hyperparameter:
            scaler = self._hyperparameter["scaler"]
            if isinstance(scaler, dict) and "__model__" in scaler and "__module__" in scaler:
                try:
                    mod = importlib.import_module(scaler["__module__"])
                    cls = getattr(mod, scaler["__model__"])
                    params = scaler.get("params", {})

                    # Convert feature_range list to tuple if needed
                    if "feature_range" in params and isinstance(params["feature_range"], list):
                        params["feature_range"] = tuple(params["feature_range"])

                    self._hyperparameter["scaler"] = cls(**params)
                except Exception as e:
                    raise ValueError(f"Error loading scaler: {e}")

    def _create_model(self):
        """
        Instantiate the LUNAR model with the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = lunar.LUNAR(**self._hyperparameter)