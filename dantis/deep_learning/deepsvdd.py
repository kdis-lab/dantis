from pyod.models import deep_svdd # deep_svdd.DeepSVDD
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)
from .. import algorithmbase
from .. import utils


class DeepSVDD(algorithmbase.AlgorithmBase):
    """
    DeepSVDD(hyperparameter: dict)
    Deep Support Vector Data Description (DeepSVDD) for unsupervised anomaly detection.
    This class implements the DeepSVDD algorithm, a deep learning-based approach for detecting anomalies in data. 
    It inherits from `algorithmbase.AlgorithmBase` and provides methods for model training, prediction, and anomaly scoring. 
    DeepSVDD learns a neural network mapping that encloses the majority of data in a hypersphere, enabling the detection of outliers as samples lying outside this region.
    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model and training hyperparameters. See `get_default_hyperparameters` for defaults.
    Attributes
    ----------
    _model : object
        The underlying DeepSVDD model instance.
    _hyperparameter : dict
        Dictionary of hyperparameters for the model.
    Methods
    -------
    decision_function(x)
        Computes the anomaly score for the input data `x` using the trained DeepSVDD model.
    fit(x_train: np.ndarray, y_train: np.ndarray = None)
        Trains the DeepSVDD model on the provided training data.
    predict(x: np.ndarray = None) -> np.ndarray
        Returns the anomaly score for the input data `x`.
    save_model(path_model)
        Saves the trained DeepSVDD model to the specified path.
    load_model(path_model)
        Loads a DeepSVDD model from the specified path.
    set_hyperparameter(hyperparameter)
        Sets the hyperparameters for the DeepSVDD model.
    _create_model()
        Internal method to build the DeepSVDD model based on the current hyperparameters.
    Notes
    -----
    - The model is designed for unsupervised anomaly detection and does not use `y_train` during training.
    - The anomaly score is typically based on the distance of samples from the center of the learned hypersphere.
    """
    def __init__(self, hyperparameter: dict):
        """
        Initialize the DeepSVDD model with the given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of model hyperparameters.
        """
        super().__init__(hyperparameter=hyperparameter)
        
        self._create_model()

    def decision_function(self, x):
        """
        Predict raw anomaly score of X using the fitted detector. 
        The anomaly score of an input sample is computed based on different detector algorithms. 
        For consistency, outliers are assigned with larger anomaly scores.
        Parameters
        ----------
        x : np.array
            Input data.

        Returns
        -------
        np.array
            The anomaly score of the input samples.        
        """
        return self._model.decision_function(x)
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Trains the DeepSVDD model with the training data.

        Parameters
        ----------
        x_train : np.array
            Training data.
        y_train : np.array, optional
            Target data (not used).

        Returns
        -------
        np.array
            Anomaly scores for the training data.
        """
        logging.info("Fitting DeepSVDD model...")
        logging.info("Don't use y_train in DeepSVDD model fitting, it is not used.")
        if "n_features" not in self._hyperparameter or self._hyperparameter["n_features"] in [None, "empty"]:
            self._hyperparameter["n_features"] = x_train.shape[1]
        else:
            self._hyperparameter["n_features"] = int(self._hyperparameter["n_features"])

        if "hidden_neurons" not in self._hyperparameter or self._hyperparameter["hidden_neurons"] is None:
            self._hyperparameter["hidden_neurons"] = [32, 16]
        else:
            self._hyperparameter["hidden_neurons"] = [int(h) for h in self._hyperparameter["hidden_neurons"]]

        self._model = deep_svdd.DeepSVDD(**self._hyperparameter)
        
        self._model.fit(x_train)

        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Computes the anomaly score for input samples `x`.

        Parameters
        ----------
        x : np.array, optional
            Input data.

        Returns
        -------
        np.array
            Anomaly scores.
        """

        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Saves the model to the specified path.

        Parameters
        ----------
        path_model : str
            Path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Loads the model from the specified path.

        Parameters
        ----------
        path_model : str
            Path to load the model from.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Sets the model hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = deep_svdd.DeepSVDD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Creates the DeepSVDD model with the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        # self._model = deep_svdd.DeepSVDD(**self._hyperparameter)