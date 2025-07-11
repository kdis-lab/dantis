from sklearn.ensemble import RandomForestClassifier as randomForest
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .. import algorithmbase
from .. import utils

class RandomForestClassifier(algorithmbase.AlgorithmBase):
    """
    Random Forest Classifier for supervised anomaly detection.

    This class wraps sklearn's RandomForestClassifier and provides a
    consistent interface with hyperparameter management and encoding
    for supervised classification tasks.

    Inherits from
    -------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary of hyperparameters to configure the Random Forest model.

    Attributes
    ----------
    _model : sklearn.ensemble.RandomForestClassifier
        The underlying sklearn RandomForestClassifier model.
    _label_encoder : sklearn.preprocessing.LabelEncoder
        Label encoder to convert string labels into numeric labels.
    _hyperparameter : dict
        Stored hyperparameters.

    Methods
    -------
    fit(x_train, y_train)
        Fit the Random Forest classifier to the training data.
    decision_function(X)
        Returns the probability of the positive class (anomaly).
    predict(x_test)
        Predicts binary labels (0 or 1) for test samples.
    save_model(path_model)
        Save the trained model to disk.
    load_model(path_model)
        Load a trained model from disk.
    set_hyperparameter(hyperparameter)
        Validate and set hyperparameters.
    _create_model()
        Instantiate the sklearn RandomForestClassifier with hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the Random Forest Classifier with specified hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters for RandomForestClassifier.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._label_encoder = LabelEncoder()
        self._create_model()        

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the probability of class 1 (anomaly) for given samples.

        Parameters
        ----------
        X : np.ndarray
            Input feature samples.

        Returns
        -------
        np.ndarray
            Probability scores of the positive class.
        """
        return self._model.predict_proba(X)[:, 1]

    def fit(self, x_train: np.ndarray = None, y_train: np.ndarray = None) -> np.ndarray:
        """
        Train the Random Forest classifier on labeled data.

        Parameters
        ----------
        x_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray
            Corresponding labels.

        Returns
        -------
        np.ndarray
            Probability scores for the training samples.
        """
        if x_train is not None and y_train is None:
            raise utils.SupervisedInputDataError("x_train is not None but y_train is None.")
        y_train_encoded = self._label_encoder.fit_transform(y_train)
        self._model.fit(x_train, y_train_encoded)
        return self.decision_function(x_train)

    def predict(self, x_test: np.ndarray = None) -> np.ndarray:
        """
        Predict binary labels for test samples based on probability threshold 0.5.

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
        Save the trained model to disk.

        Parameters
        ----------
        path_model : str
            File path where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load a trained model from disk.

        Parameters
        ----------
        path_model : str
            File path from which to load the model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Set and validate hyperparameters for the Random Forest classifier.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = randomForest
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the Random Forest classifier with the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = randomForest(**self._hyperparameter)