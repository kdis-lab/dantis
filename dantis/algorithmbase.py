import numpy as np
import joblib
import logging

class AlgorithmBase:
    """
    Abstract base class for anomaly detection algorithms.

    This class defines a standard interface for training, scoring, saving,
    and loading anomaly detection models. It must be subclassed, and the
    methods `fit`, `predict`, and `decision_function` must be implemented.

    Attributes
    ----------
    _model : object or None
        Internal model object, assigned by the implementing subclass.
    _hyperparameter : dict or None
        Dictionary of model hyperparameters used during instantiation or training.
    check_hyperparams : bool
        Flag indicating whether to perform hyperparameter validation (default: False).

    Parameters
    ----------
    hyperparameter : dict
        Dictionary containing model-specific hyperparameters.

    Methods
    -------
    set_hyperparameter(hyperparameter)
        Set or update the hyperparameter dictionary.
    
    get_hyperparameter()
        Retrieve the current set of hyperparameters.
    
    fit(x_train, y_train=None)
        Train the model. Must be implemented by subclasses.
    
    predict(x)
        Return binary predictions. Must be implemented by subclasses if applicable.
    
    decision_function(x)
        Return continuous anomaly scores. Must be implemented by subclasses.
    
    get_anomaly_score(x)
        Alias for `decision_function(x)`; provides uniform access to anomaly scores.
    
    save_model(path_model)
        Save the model object to disk using joblib.
    
    load_model(path_model)
        Load a model object from a `.joblib` file.
    """

    _model = None
    _hyperparameter = None
    check_hyperparams = False

    def __init__(self, hyperparameter: dict) -> None:
        """
        Initialize the AlgorithmBase with given hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            A dictionary of hyperparameters specific to the subclassed model.
        """
        self._hyperparameter = hyperparameter

    def set_hyperparameter(self, hyperparameter):
        """
        Set the model's hyperparameters.

        Parameters
        ----------
        hyperparameter : dict
            New hyperparameters to assign. If None, existing ones remain unchanged.
        """
        self._hyperparameter = hyperparameter if hyperparameter is not None else self._hyperparameter

    def get_hyperparameter(self):
        """
        Retrieve the model's current hyperparameters.

        Returns
        -------
        dict
            Dictionary containing the currently set hyperparameters.
        """
        return self._hyperparameter

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        Fit the model to training data.

        This is an abstract method and must be implemented by subclasses.

        Parameters
        ----------
        x_train : np.ndarray
            Training input features, shape (n_samples, n_features).
        y_train : np.ndarray, optional
            Optional target labels (used in semi-supervised models).

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement `fit()`.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Return binary anomaly predictions.

        This is an abstract method and must be implemented by subclasses if applicable.

        Parameters
        ----------
        x : np.ndarray
            Input samples to classify, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Binary array indicating anomaly labels (1 for anomaly, 0 for normal).

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement `predict()` if applicable.")

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the anomaly score for each input sample.

        This is an abstract method and must be implemented by subclasses.

        Parameters
        ----------
        x : np.ndarray
            Input data, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Anomaly scores as continuous values, higher indicates more anomalous.

        Raises
        ------
        NotImplementedError
            If the method is not implemented in the subclass.
        """
        raise NotImplementedError("Subclasses must implement `decision_function(X)`.")

    def get_anomaly_score(self, x: np.ndarray) -> np.ndarray:
        """
        Return the anomaly scores for the input samples.

        This is a standardized interface that delegates to `decision_function`.

        Parameters
        ----------
        x : np.ndarray
            Input data, shape (n_samples, n_features).

        Returns
        -------
        np.ndarray
            Anomaly scores.
        """
        if len(np.unique(x)) == 1:
            logging.warning("All values are equals, return the same input")
            return x
        return self.decision_function(x)

    def save_model(self, path_model: str):
        """
        Save the trained model to disk in `.joblib` format.

        If the given path does not have a `.joblib` extension, it will be corrected.

        Parameters
        ----------
        path_model : str
            Path to the file where the model will be saved.
        """
        if "." in path_model:
            extension = path_model[path_model.rfind(".")+1:]
            if extension != "joblib":
                print("Warning: Change extension")
                extension = "joblib"
        path_model = path_model[:path_model.rfind(".")+1] + extension
        joblib.dump(self._model, path_model)

    def load_model(self, path_model: str):
        """
        Load a model from a `.joblib` file.

        Parameters
        ----------
        path_model : str
            Path to the file from which to load the model.

        Raises
        ------
        Exception
            If the file extension is not `.joblib`.
        """
        if "." in path_model:
            extension = path_model[path_model.rfind(".")+1:]
            if extension != "joblib":
                raise Exception("Error: extension required .joblib")
        self._model = joblib.load(path_model)
    