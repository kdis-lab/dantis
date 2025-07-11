from sklearn.neural_network import MLPClassifier as mlp
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .. import algorithmbase
from .. import utils

class MLPClassifier(algorithmbase.AlgorithmBase):
    """
    MLPClassifier(hyperparameter: dict)

    A supervised anomaly detection classifier using a Multi-layer Perceptron (MLP).
    This class wraps the scikit-learn MLPClassifier.

    Inherits from
    -------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing hyperparameters for the MLP model.

    Attributes
    ----------
    _model : mlp
        The underlying scikit-learn MLPClassifier instance.
    _hyperparameter : dict
        Dictionary storing the model's hyperparameters.
    _label_encoder : LabelEncoder
        Used to encode labels to numeric values for training.
    
    Methods
    -------
    decision_function(X)
        Returns probability scores for class 1 (anomaly).
    fit(x_train, y_train)
        Trains the MLP model on labeled training data.
    predict(x_test)
        Predicts binary anomaly labels based on a threshold.
    save_model(path_model)
        Saves the trained model to a file.
    load_model(path_model)
        Loads a trained model from a file.
    set_hyperparameter(hyperparameter)
        Sets and validates model hyperparameters.
    _create_model()
        Instantiates the model using current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the MLPClassifier with the given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Configuration dictionary for model initialization.
        """
        super().__init__(hyperparameter=hyperparameter)
        self._label_encoder = LabelEncoder()
        self._create_model()

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Return the probability of each sample being an anomaly (class 1).

        Parameters
        ----------
        X : np.ndarray
            Input feature matrix.

        Returns
        -------
        np.ndarray
            Probability estimates for class 1.
        """
        return self._model.predict_proba(X)[:, 1]

    def fit(self, x_train: np.ndarray = None, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the MLP model on labeled training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training feature matrix.
        y_train : np.ndarray
            Training labels (binary classification: 0 = normal, 1 = anomaly).

        Returns
        -------
        np.ndarray
            Anomaly scores (probabilities) for training data.

        Raises
        ------
        utils.SupervisedInputDataError
            If x_train is provided but y_train is None.
        """
        if x_train is not None and y_train is None:
            raise utils.SupervisedInputDataError("x_train is not None but y_train is None.")
        
        y_train_encoded = self._label_encoder.fit_transform(y_train)
        self._model.fit(x_train, y_train_encoded)

        return self.decision_function(x_train)

    def predict(self, x_test: np.ndarray = None) -> np.ndarray:
        """
        Predict binary labels (0 = inlier, 1 = outlier) based on decision threshold.

        Parameters
        ----------
        x_test : np.ndarray
            Test samples.

        Returns
        -------
        np.ndarray
            Binary predictions for each input sample.
        """
        scores = self.decision_function(x_test)
        return (scores >= 0.5).astype(int)

    def save_model(self, path_model: str):
        """
        Save the trained MLP model to the specified path.

        Parameters
        ----------
        path_model : str
            File path to save the model.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load a trained MLP model from the specified file.

        Parameters
        ----------
        path_model : str
            File path of the saved model.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Set and validate the hyperparameters of the MLP model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary containing MLP hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = mlp  # class reference
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the MLP model using the current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = mlp(**self._hyperparameter)