from pyod.models import cd # cd.CD
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression

from .. import utils
from .. import algorithmbase
import importlib

logging.basicConfig(level=logging.INFO)

class CD(algorithmbase.AlgorithmBase):
    """
    CD (Copula-Based Outlier Detection) anomaly detection model.

    This class wraps the PyOD CD model, which detects outliers based on
    copula modeling.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing the hyperparameters for the CD algorithm.

    Methods
    -------
    fit(x_train, y_train=None)
        Fits the CD model using the training data.
    decision_function(x)
        Returns the anomaly scores for the samples.
    predict(x)
        Predicts outlier probabilities or scores for given samples.
    save_model(path_model)
        Saves the trained model to disk.
    load_model(path_model)
        Loads the model from disk.
    set_hyperparameter(hyperparameter)
        Validates and sets hyperparameters.
    _create_model()
        Instantiates the CD model with current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the CD model.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for the CD algorithm.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._create_model()

    def decision_function(self, x):
        """
        Compute anomaly scores for the given samples.

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
        Fit the CD model on the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training samples.
        y_train : np.ndarray, optional
            Not used but accepted for interface consistency.

        Returns
        -------
        np.ndarray
            Outlier scores in [0,1] for training samples.
        """
        logging.info("Fitting CD model...")
        logging.info("Note: y_train is not used in CD fitting.")
        if x_train.shape[1] < 2:
            raise ValueError(f"CD requires at least 2 features. Provided: {x_train.shape[1]}.")
        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict outlier scores for the given samples.

        Parameters
        ----------
        x : np.ndarray
            Samples to predict.

        Returns
        -------
        np.ndarray
            Predicted outlier probabilities or scores.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the CD model to a file.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the CD model from a file.

        Parameters
        ----------
        path_model : str
            File path from where the model is loaded.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Validate and set hyperparameters for the CD model.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters dictionary.
        """
        self._hyperparameter = hyperparameter
        self._model = cd.CD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the CD model with current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)

        model = self._hyperparameter.get("model")

        # Deserialize if model is a dict
        if isinstance(model, dict) and "__model__" in model and "__module__" in model:
            self._hyperparameter["model"] = _deserialize_model(model)
        elif not hasattr(model, "fit"):
            logging.warning("Invalid 'model' parameter; using LogisticRegression by default.")
            self._hyperparameter["model"] = LogisticRegression()

        self._model = cd.CD(**self._hyperparameter)


def _deserialize_model(model_dict):
    """
    Deserialize a sklearn estimator from a dictionary.

    Parameters
    ----------
    model_dict : dict
        Serialized model dictionary.

    Returns
    -------
    sklearn estimator
        The deserialized sklearn model instance.
    """
    try:
        class_name = model_dict['__model__']
        module_name = model_dict['__module__']
        params = model_dict.get('params', {})

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)

        return model_class(**params)
    except Exception as e:
        logging.warning(f"Failed to deserialize model from dict: {e}. Using LogisticRegression by default.")
        return LogisticRegression()