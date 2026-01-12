from sklearn.covariance import MinCovDet
import numpy as np
import logging

from .. import algorithmbase

logging.basicConfig(level=logging.INFO)

class FastMCD(algorithmbase.AlgorithmBase):
    """
    FastMCD(hyperparameter: dict)

    Fast Minimum Covariance Determinant (FastMCD) for anomaly detection.

    This class wraps the sklearn implementation of MinCovDet. It detects anomalies
    by computing the robust covariance of the data and using the Mahalanobis 
    distance as the anomaly score.

    Inherits from
    --------------
    algorithmbase.AlgorithmBase

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model configuration values.

    Attributes
    ----------
    _model : sklearn.covariance.MinCovDet
        Instance of the sklearn MinCovDet model.
    _hyperparameter : dict
        Dictionary of model hyperparameters.

    Methods
    -------
    decision_function(x)
        Returns the Mahalanobis distance for input data (anomaly score).
    fit(x_train, y_train=None)
        Fits the MinCovDet model using training data.
    predict(x)
        Returns anomaly scores (consistent with provided examples).
    save_model(path_model)
        Saves the trained model to a specified file path.
    load_model(path_model)
        Loads the model from a specified file.
    """

    def __init__(self, hyperparameter: dict):
        """
        Initializes the FastMCD model with the provided hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            Dictionary containing model parameters.
            Supported keys:
            - store_precision (bool, default=True)
            - support_fraction (float, default=None)
            - random_state (int, default=42)
            - assume_centered (bool, default=False)
        """
        super().__init__(hyperparameter=hyperparameter)
        
        # Valores por defecto basados en algorithm.py y sklearn
        self.store_precision = self._hyperparameter.get('store_precision', True)
        self.support_fraction = self._hyperparameter.get('support_fraction', None)
        self.random_state = self._hyperparameter.get('random_state', 42)
        self.assume_centered = self._hyperparameter.get('assume_centered', False)
        
        self._create_model()

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the Mahalanobis distance for the input data.

        Parameters
        ----------
        x : np.ndarray
            Input data for scoring.

        Returns
        -------
        np.ndarray
            Mahalanobis distances. Higher values indicate a higher likelihood of being an outlier.
        """
        # MinCovDet calcula la distancia de Mahalanobis
        return self._model.mahalanobis(x)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None) -> np.ndarray:
        """
        Fits the FastMCD (MinCovDet) model to the training data.

        Parameters
        ----------
        x_train : np.ndarray
            Feature matrix for training.
        y_train : np.ndarray, optional
            Target labels (not used).

        Returns
        -------
        np.ndarray
            Anomaly scores for the training data.
        """
        logging.info("Fitting FastMCD model...")
        self._model.fit(x_train)

        return self.get_anomaly_score(x_train)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predicts anomaly scores for the input data.
        
        Note: Consistent with other implementations in this library, 
        this returns the raw scores rather than binary labels.

        Parameters
        ----------
        x : np.ndarray
            Input data.

        Returns
        -------
        np.ndarray
            Anomaly scores.
        """
        return self.get_anomaly_score(x)

    def _create_model(self):
        """
        Instantiates the MinCovDet model using the current hyperparameters.
        """
        self._model = MinCovDet(
            store_precision=self.store_precision,
            assume_centered=self.assume_centered,
            support_fraction=self.support_fraction,
            random_state=self.random_state
        )