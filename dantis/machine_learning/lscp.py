from pyod.models import lscp # lscp.LSCP
import numpy as np
import logging

import importlib
from .. import algorithmbase
from .. import utils

logging.basicConfig(level=logging.INFO)

class LSCP(algorithmbase.AlgorithmBase):
    """
    LSCP anomaly detection model wrapper.

    Parameters
    ----------
    hyperparameter : dict
        Hyperparameters for the LSCP algorithm, must include 'detector_list'.

    Methods
    -------
    fit(x_train, y_train=None)
        Fit the LSCP model on training data.
    decision_function(x)
        Compute anomaly scores for input data.
    predict(x)
        Predict outliers for given samples.
    save_model(path_model)
        Save model to disk.
    load_model(path_model)
        Load model from disk.
    set_hyperparameter(hyperparameter)
        Validate and set hyperparameters including detector_list.
    _create_model()
        Instantiate the LSCP model.
    """

    @classmethod
    def get_default_hyperparameters(cls) -> dict:
        try:
            from pyod.models.lof import LOF
            default_detectors = [LOF(), LOF()]
        except Exception:
            default_detectors = []
        return {
            "detector_list": default_detectors,
            "contamination": 0.1,
        }

    def __init__(self, hyperparameter: dict | None = None):
        """Initialize the LSCP detector.

        Do not create the underlying PyOD model at construction time. This
        avoids raising errors when tooling (like model discovery) instantiates
        the class with an empty or partial hyperparameter dict.
        """
        hp = hyperparameter if hyperparameter is not None else self.get_default_hyperparameters()
        super().__init__(hyperparameter=hp)
        self.check_hyperparams = True
        self._model = None

    def decision_function(self, x: np.ndarray):
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
        Fit the LSCP model using the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training data.
        y_train : np.ndarray, optional
            Not used.

        Returns
        -------
        np.ndarray
            Outlier scores of the training data [0,1].
        """
        self._model.fit(x_train)
        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict whether samples are outliers.

        Parameters
        ----------
        x : np.ndarray, optional
            Samples to predict.

        Returns
        -------
        np.ndarray
            Outlier prediction probabilities.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save the LSCP model to a file.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load the LSCP model from a file.

        Parameters
        ----------
        path_model : str
            File path to load the model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Validate and set the hyperparameters.

        Checks and loads detector_list to proper instances.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters dictionary, must include 'detector_list'.
        """
        self._hyperparameter = hyperparameter
        self._model = lscp.LSCP

        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

        if "detector_list" not in self._hyperparameter:
            raise utils.SupervisedInputDataError("detector_list param is needed")

        if not hasattr(self._hyperparameter["detector_list"], "__iter__"):
            if self.check_hyperparams:
                from pyod.models.lof import LOF
                self._hyperparameter["detector_list"] = [LOF(), LOF()]
            else:
                raise utils.SupervisedInputDataError("LSCP needs a list with the ad_algorithms to be used")

        detector_list = self._hyperparameter["detector_list"]

        new_list = []
        for det in detector_list:
            if isinstance(det, dict) and "__model__" in det and "__module__" in det:
                try:
                    mod = importlib.import_module(det["__module__"])
                    cls = getattr(mod, det["__model__"])
                    instance = cls(**det.get("params", {}))
                    new_list.append(instance)
                except Exception as e:
                    raise ValueError(f"Error loading detector {det}: {e}")
            else:
                new_list.append(det)

        self._hyperparameter["detector_list"] = new_list

        if not all(hasattr(d, "fit") for d in new_list):
            raise utils.SupervisedInputDataError("All detectors must be valid PyOD instances")

    def _create_model(self):
        """
        Instantiate the LSCP model using the set hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = lscp.LSCP(**self._hyperparameter)