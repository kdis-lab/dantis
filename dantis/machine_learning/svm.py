from sklearn.svm import SVC
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .. import algorithmbase
from .. import utils

class SVM(algorithmbase.AlgorithmBase):
    """
    Support Vector Machine (SVM) classifier for supervised anomaly detection.

    This class wraps sklearn's SVC with probability estimates enabled, 
    providing a consistent interface with hyperparameter management and label encoding.

    Inherits from
    -------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary of hyperparameters for configuring the SVM model.

    Attributes
    ----------
    _model : sklearn.svm.SVC
        The underlying SVM classifier model.
    _label_encoder : sklearn.preprocessing.LabelEncoder
        Encoder to convert string labels to numeric labels.
    _hyperparameter : dict
        Stored hyperparameters for the model.

    Methods
    -------
    fit(x_train, y_train)
        Fit the SVM classifier to training data.
    decision_function(X)
        Returns the probability of the positive class (anomaly).
    predict(x_test)
        Predict binary anomaly labels (0 or 1) for test samples.
    save_model(path_model)
        Save the trained model to a file.
    load_model(path_model)
        Load a trained model from a file.
    set_hyperparameter(hyperparameter)
        Validate and set hyperparameters.
    _create_model()
        Instantiate the SVM model with the current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the SVM classifier with specified hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for sklearn.svm.SVC.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._label_encoder = LabelEncoder()
        self._create_model()        

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the probability of class 1 (anomaly) for the given samples.

        Parameters
        ----------
        X : np.ndarray
            Feature samples.

        Returns
        -------
        np.ndarray
            Probability scores of the positive class.
        """
        return self._model.predict_proba(X)[:, 1]

    def fit(self, x_train: np.ndarray = None, y_train: np.ndarray = None) -> np.ndarray:
        """
        Train the SVM model on labeled training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Corresponding training labels.

        Returns
        -------
        np.ndarray
            Probability scores for training samples.
        """
        if x_train is not None and y_train is None:
            raise utils.SupervisedInputDataError("x_train is not None but y_train is None.")
        y_train_encoded = self._label_encoder.fit_transform(y_train)
        self._model.fit(x_train, y_train_encoded)
        return self.decision_function(x_train)

    def predict(self, x_test: np.ndarray = None) -> np.ndarray:
        """
        Predict binary anomaly labels for test samples using probability threshold 0.5.

        Parameters
        ----------
        x_test : np.ndarray
            Feature matrix for testing.

        Returns
        -------
        np.ndarray
            Predicted labels (0 or 1).
        """
        scores = self.decision_function(x_test)
        return (scores >= 0.5).astype(int)

    def save_model(self, path_model):
        """
        Save the trained SVM model to disk.

        Parameters
        ----------
        path_model : str
            File path where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load a trained SVM model from disk.

        Parameters
        ----------
        path_model : str
            File path from which to load the model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Set and validate hyperparameters for the SVM model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = SVC
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the SVM model with probability enabled and current hyperparameters.
        """
        self._hyperparameter["probability"] = True
        self.set_hyperparameter(self._hyperparameter)
        self._model = SVC(**self._hyperparameter)