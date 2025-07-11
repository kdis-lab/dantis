from pyod.models import alad # alad.ALAD
import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from .. import algorithmbase
from .. import utils

class ALAD(algorithmbase.AlgorithmBase):
    """
    ALAD(hyperparameter: dict)
    Adversarially Learned Anomaly Detection (ALAD) for unsupervised anomaly detection.
    This class implements the ALAD algorithm, an adversarial approach for detecting anomalies in data. 
    It inherits from `algorithmbase.AlgorithmBase` and provides methods for model training, prediction, and anomaly scoring. 
    The ALAD model is based on adversarial learning, where a generator and discriminator are trained to distinguish between normal and anomalous samples.
    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model and training hyperparameters. See `get_default_hyperparameters` for defaults.
    Attributes
    ----------
    _model : object
        The underlying ALAD model instance.
    _hyperparameter : dict
        Dictionary of hyperparameters for the model.
    Methods
    -------
    decision_function(x)
        Computes the anomaly score for the input data `x` using the trained ALAD model.
    fit(x_train: np.ndarray, y_train: np.ndarray = None)
        Trains the ALAD model on the provided training data.
    predict(x: np.ndarray = None) -> np.ndarray
        Returns the anomaly score for the input data `x`.
    save_model(path_model)
        Saves the trained ALAD model to the specified path.
    load_model(path_model)
        Loads an ALAD model from the specified path.
    set_hyperparameter(hyperparameter)
        Sets the hyperparameters for the ALAD model.
    _create_model()
        Internal method to build the ALAD model based on the current hyperparameters.
    Notes
    -----
    - The model is designed for unsupervised anomaly detection and does not use `y_train` during training.
    - The anomaly score is typically based on the output of the adversarial network.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the ALAD model with the given hyperparameters.

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
    
    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        Trains the alad model with the training data.

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
        logging.info("Fitting ALAD model...")
        logging.info("Don't use y_train in ALAD model fitting, it is not used.")
        hyperparameter = self.get_hyperparameter()
        noise_std = 0.1 if "noise_std" not in hyperparameter.keys() else hyperparameter["noise_std"]
        self._model.fit(x_train, noise_std=noise_std)
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
        self._model = alad.ALAD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Creates the ALAD model with the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = alad.ALAD(**self._hyperparameter)