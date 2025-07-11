from sklearn.naive_bayes import GaussianNB as naiveBayes
import numpy as np
from sklearn.preprocessing import LabelEncoder

from .. import algorithmbase
from .. import utils

class GaussianNB(algorithmbase.AlgorithmBase):
    """
    GaussianNB(hyperparameter: dict)

    A supervised anomaly detection classifier using Gaussian Naive Bayes.
    This wrapper handles continuous input features assuming a Gaussian distribution per feature.

    Inherits from
    -------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing hyperparameters for the GaussianNB model.

    Attributes
    ----------
    _model : GaussianNB
        Instance of scikit-learn's GaussianNB.
    _hyperparameter : dict
        Stored hyperparameters used for model initialization.
    _label_encoder : LabelEncoder
        Encodes class labels into integers.

    Methods
    -------
    fit(x_train, y_train)
        Fits the model to the training data.
    decision_function(X)
        Returns probability estimates for class 1 (anomaly).
    predict(x_test)
        Predicts binary anomaly labels.
    save_model(path_model)
        Saves the model to the specified path.
    load_model(path_model)
        Loads a previously saved model.
    set_hyperparameter(hyperparameter)
        Sets and validates model hyperparameters.
    _create_model()
        Instantiates the model using the current hyperparameters.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initialize the GaussianNB model with the given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Configuration dictionary for the model.
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
            Input samples.

        Returns
        -------
        np.ndarray
            Probability estimates for class 1.
        """
        return self._model.predict_proba(X)[:, 1]

    def fit(self, x_train: np.ndarray = None, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fit the model using the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training labels (binary).

        Returns
        -------
        np.ndarray
            Probability scores for training data.

        Raises
        ------
        utils.SupervisedInputDataError
            If x_train is provided but y_train is missing.
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
            Binary predictions based on 0.5 threshold.
        """
        scores = self.decision_function(x_test)
        return (scores >= 0.5).astype(int)

    def save_model(self, path_model: str):
        """
        Save the trained GaussianNB model to disk.

        Parameters
        ----------
        path_model : str
            File path where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model: str):
        """
        Load a GaussianNB model from a saved file.

        Parameters
        ----------
        path_model : str
            Path to the saved model file.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter: dict):
        """
        Validate and set the hyperparameters for the model.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary of hyperparameters.
        """
        self._hyperparameter = hyperparameter
        self._model = naiveBayes
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
        Instantiate the GaussianNB model using current hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = naiveBayes(**self._hyperparameter)