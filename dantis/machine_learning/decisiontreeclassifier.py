from sklearn.tree import DecisionTreeClassifier as SklearnDTC
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .. import algorithmbase
from .. import utils

class DecisionTreeClassifier(algorithmbase.AlgorithmBase):
    """
    DecisionTreeClassifier(hyperparameter: dict)

    A supervised anomaly detection model based on Scikit-learn's DecisionTreeClassifier.

    This class inherits from AlgorithmBase and wraps a decision tree classifier to perform binary
    anomaly detection, where class 1 is considered an anomaly. The model outputs anomaly scores
    based on the predicted probability of class 1.

    Parameters
    ----------
    hyperparameter : dict
        Dictionary of hyperparameters used to configure the DecisionTreeClassifier.

    Attributes
    ----------
    _model : SklearnDTC
        Instance of the underlying scikit-learn DecisionTreeClassifier.
    _label_encoder : LabelEncoder
        Used to convert string labels to integers during training.
    _hyperparameter : dict
        Hyperparameter dictionary passed during initialization.

    Methods
    -------
    decision_function(X: np.ndarray) -> np.ndarray
        Returns the probability of each sample being an anomaly (class 1).

    fit(x_train: np.ndarray, y_train: np.ndarray) -> np.ndarray
        Fits the model using the training data and returns anomaly scores.

    predict(x_test: np.ndarray) -> np.ndarray
        Predicts binary anomaly labels (1 for anomaly, 0 for normal).

    save_model(path_model)
        Saves the trained model to disk.

    load_model(path_model)
        Loads a model from a previously saved state.

    set_hyperparameter(hyperparameter: dict)
        Sets or updates the model hyperparameters.

    _create_model()
        Internal method to instantiate or update the DecisionTreeClassifier with current parameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initializes the DecisionTreeClassifier.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary containing hyperparameters for the classifier.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._label_encoder = LabelEncoder()
        self._create_model()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Returns anomaly scores for input samples based on probability of being class 1.

        Parameters
        ----------
        X : np.ndarray
            Input samples.

        Returns
        -------
        np.ndarray
            Probability scores indicating likelihood of being an anomaly.
        """
        return self._model.predict_proba(X)[:, 1]

    def fit(self, x_train: np.ndarray = None, y_train: np.ndarray = None) -> np.ndarray:
        """
        Trains the DecisionTreeClassifier model on the given labeled dataset.

        Parameters
        ----------
        x_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels (0 for normal, 1 for anomaly).

        Returns
        -------
        np.ndarray
            Anomaly scores for the training data.

        Raises
        ------
        SupervisedInputDataError
            If x_train is provided but y_train is None.
        """
        if x_train is not None and y_train is None:
            raise utils.SupervisedInputDataError("x_train is provided but y_train is None")

        y_train_encoded = self._label_encoder.fit_transform(y_train)
        self._model.fit(x_train, y_train_encoded)

        return self.decision_function(x_train)

    def predict(self, x_test: np.ndarray = None) -> np.ndarray:
        """
        Predicts binary anomaly labels for the input samples.

        Parameters
        ----------
        x_test : np.ndarray
            Test data to classify.

        Returns
        -------
        np.ndarray
            Array with binary predictions: 1 (anomaly) or 0 (normal).
        """
        scores = self.decision_function(x_test)
        return (scores >= 0.5).astype(int)

    def save_model(self, path_model):
        """
        Saves the trained model to the specified path.

        Parameters
        ----------
        path_model : str
            Path where the model should be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Loads a previously saved model from the specified path.

        Parameters
        ----------
        path_model : str
            Path to the saved model file.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Sets the model's hyperparameters and updates the model accordingly.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters to set.
        """
        self._hyperparameter = hyperparameter
        parameters_management = utils.ParameterManagement(SklearnDTC.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

        if hasattr(self, "_model") and isinstance(self._model, SklearnDTC):
            self._model.set_params(**self._hyperparameter)
        else:
            self._model = SklearnDTC(**self._hyperparameter)

    def _create_model(self):
        """
        Internal method that creates the DecisionTreeClassifier instance using the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)