import numpy as np
import logging
logging.basicConfig(level=logging.INFO)

from .. import algorithmbase

class RandomBaseline(algorithmbase.AlgorithmBase):
    """
    Random baseline anomaly scorer that produces random scores in [0, 1].

    This baseline generates independent uniform random anomaly scores for each
    sample. A `random_state` hyperparameter controls the RNG seed to make the
    outputs reproducible.

    Parameters
    ----------
    hyperparameter : dict, optional
        Dictionary of model hyperparameters. Supported keys:
        - "random_state" : int
            Seed for the random number generator (default: 42).

    Attributes
    ----------
    _train_length : int
        Length of the training series stored during `fit`.
    _random_state : int
        Seed used to initialize the random number generator.
    """

    def __init__(self, hyperparameter: dict = None):
        """
        Initialize the RandomBaseline.

        Parameters
        ----------
        hyperparameter : dict, optional
            Hyperparameters for the model (kept for compatibility).
        """
        super().__init__(hyperparameter=hyperparameter)
        self._train_length = 0
        # default seed
        self._random_state = 42
        # apply provided hyperparameters if any
        if hyperparameter:
            self.set_hyperparameter(hyperparameter)

    def decision_function(self, x):
        """
        Compute random anomaly scores for the input data.

        The function returns a 1-D array of scores sampled uniformly in [0, 1].
        If `x` is None, the method uses the length saved during `fit`.

        Parameters
        ----------
        x : array-like or None
            Input data whose first dimension defines the number of samples.
            If None, the stored training length is used.

        Returns
        -------
        numpy.ndarray
            1-D array of anomaly scores with shape (n_samples,). Returns an
            empty array if `n_samples` is 0.
        """
        if x is None:
            n = getattr(self, "_train_length", 0)
        else:
            try:
                n = int(np.asarray(x).shape[0])
            except Exception:
                n = 0

        if n <= 0:
            return np.array([])

        # Use a reproducible RNG seeded with _random_state
        rng = np.random.RandomState(self._random_state)
        scores = rng.uniform(0.0, 1.0, size=n).astype(float).reshape(-1)
        return scores

    def fit(self, x_train: np.ndarray, y_train: np.ndarray = None):
        """
        "Fit" the baseline by recording the training series length.

        No learning is performed. Only the number of samples in `x_train`
        is recorded so that `decision_function(None)` can generate scores
        for that length.

        Parameters
        ----------
        x_train : numpy.ndarray
            Training data. Only its length (first dimension) is used.
        y_train : numpy.ndarray, optional
            Unused. Present for API compatibility.

        Returns
        -------
        numpy.ndarray
            Anomaly scores for `x_train`.
        """
        logging.info("Fitting RandomBaseline (no-op). y_train ignored.")
        self._train_length = 0 if x_train is None else int(np.asarray(x_train).shape[0])
        return self.get_anomaly_score(x_train)

    def get_anomaly_score(self, x: np.ndarray = None):
        """
        Alias for `decision_function` to comply with the expected API.

        Parameters
        ----------
        x : numpy.ndarray or None, optional
            Input data or None to use recorded training length.

        Returns
        -------
        numpy.ndarray
            Anomaly scores produced by `decision_function`.
        """
        return self.decision_function(x)

    def predict(self, x: np.ndarray = None) -> np.ndarray:
        """
        Predict anomaly scores for the given input.

        Parameters
        ----------
        x : numpy.ndarray or None, optional
            Input data.

        Returns
        -------
        numpy.ndarray
            Anomaly scores.
        """
        return self.get_anomaly_score(x)

    def save_model(self, path_model):
        """
        Save model state to the given path.

        For this baseline there is no complex state to serialize beyond what
        the base class may handle.

        Parameters
        ----------
        path_model : str
            Path where the model should be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
        Load model state from the given path.

        Parameters
        ----------
        path_model : str
            Path from which the model should be loaded.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
        Set hyperparameters for compatibility with the base API.

        Supported hyperparameters:
        - "random_state" : int, optional
            Seed for the RNG. If not provided, keeps the current seed
            (default 42).

        Parameters
        ----------
        hyperparameter : dict
            Hyperparameters to set.
        """
        if not hyperparameter:
            return
        # extract random_state if present, otherwise keep existing
        if isinstance(hyperparameter, dict) and "random_state" in hyperparameter:
            try:
                self._random_state = int(hyperparameter["random_state"])
            except Exception:
                logging.warning("Invalid random_state provided; keeping previous seed.")
        # keep a reference to the hyperparameter dict for compatibility
        self._hyperparameter = hyperparameter or {}