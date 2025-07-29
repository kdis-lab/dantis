def holdout_split(X, y, val_size=0.1, test_size=0.2):
    """
    Perform a holdout split on time series or tabular data.

    Splits the dataset into train, validation, and test sets. If `val_size` is 0,
    only train and test sets are returned with validation set as `None`.

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like or None
        Target values corresponding to X.
    val_size : float, optional
        Proportion of the training set to allocate to validation. Default is 0.1.
    test_size : float, optional
        Proportion of the dataset to allocate to test. Default is 0.2.

    Returns
    -------
    tuple
        A 3-tuple containing:
        - (X_train, y_train): Training set
        - (X_val, y_val): Validation set (or (None, None) if val_size=0)
        - (X_test, y_test): Test set
    """
    n = len(X)
    test_split = int(n * (1 - test_size))

    if val_size == 0: 
        X_train, X_test = X[:test_split], X[test_split:]
        y_train = y_test = None
        if y is not None:
            y_train, y_test = y[:test_split], y[test_split:]
        
        return (X_train, y_train), (None, None), (X_test, y_test)

    else: 
        val_split = int(test_split * (1 - val_size))
        X_train, X_val, X_test = X[:val_split], X[val_split:test_split], X[test_split:]
        y_train = y_val = y_test = None
        if y is not None:
            y_train, y_val, y_test = y[:val_split], y[val_split:test_split], y[test_split:]
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
def expanding_split(X, y, k=5, test_size=0.2):
    """
    Perform an expanding window time series validation split.

    The training window grows at each step, while the validation set slides forward.

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like or None
        Target values corresponding to X.
    k : int, optional
        Number of expanding validation folds. Default is 5.
    test_size : float, optional
        Proportion of data to allocate for the final test set. Default is 0.2.

    Returns
    -------
    tuple
        A 2-tuple:
        - folds : list of tuples
            List of (train, val) pairs for expanding folds
        - (X_test, y_test) : tuple
            Test set after expanding validation
    """
    n = len(X)
    test_split = int(n * (1 - test_size))
    folds = []

    step_size = test_split // (k + 1)

    for i in range(1, k + 1):
        end_train = i * step_size
        end_val = end_train + step_size
        if end_val >= test_split:
            break

        X_train, X_val = X[:end_train], X[end_train:end_val]
        y_train = y_val = None
        if y is not None:
            y_train, y_val = y[:end_train], y[end_train:end_val]

        folds.append(((X_train, y_train), (X_val, y_val)))

    X_test = X[test_split:]
    y_test = y[test_split:] if y is not None else None
    return folds, (X_test, y_test)

def sliding_split(X, y, window_size=0.5, test_size=0.2):
    """
    Perform a sliding window time series validation split.

    A fixed-size training window slides forward with each fold, optionally overlapping.

    Parameters
    ----------
    X : array-like
        Input features.
    y : array-like or None
        Target values corresponding to X.
    window_size : float, optional
        Proportion of training data used for each training window. Default is 0.5.
    test_size : float, optional
        Proportion of the dataset reserved for test set. Default is 0.2.

    Returns
    -------
    tuple
        A 2-tuple:
        - folds : list of tuples
            List of (train, val) pairs for sliding windows
        - (X_test, y_test) : tuple
            Test set after sliding validation
    """
    n = len(X)
    test_split = int(n * (1 - test_size))
    folds = []

    window_len = int(test_split * window_size)
    step_size = int(window_len * 0.5) 

    start = 0
    while True:
        end_train = start + window_len
        end_val = end_train + step_size

        if end_val >= test_split:
            break

        X_train, X_val = X[start:end_train], X[end_train:end_val]
        y_train = y_val = None
        if y is not None:
            y_train, y_val = y[start:end_train], y[end_train:end_val]

        folds.append(((X_train, y_train), (X_val, y_val)))
        start += step_size

    X_test = X[test_split:]
    y_test = y[test_split:] if y is not None else None
    return folds, (X_test, y_test)

class ValidationController: 
    """
    Controller for storing and retrieving validation configurations.

    Stores a general-purpose configuration dictionary and auxiliary column information.

    Attributes
    ----------
    validation_config : dict
        Dictionary storing validation configuration settings.

    info_columns : dict
        Dictionary holding metadata or info for display in UI or logging.

    Methods
    -------
    set_validation_config(validation_config)
        Set the validation configuration dictionary.

    get_validation_config()
        Retrieve the current validation configuration.

    set_info_tabla(info_columns)
        Store additional column-related metadata.

    get_info_tabla()
        Retrieve stored column information.
    """
    def __init__(self):
        """
        Initialize the ValidationController with empty configuration and column info.

        Attributes
        ----------
        validation_config : dict
            Initialized as an empty dictionary to store validation settings.

        info_columns : dict
            Initialized as an empty dictionary to store auxiliary column metadata.
        """
        self.validation_config = {}
        self.info_columns = {}

    def set_validation_config (self, validation_config): 
        """
        Set the validation configuration dictionary.

        Parameters
        ----------
        validation_config : dict
            Configuration settings for validation strategies.
        """
        self.validation_config =  validation_config

    def get_validation_config (self): 
        """
        Retrieve the current validation configuration.

        Returns
        -------
        dict
            The validation configuration dictionary.
        """
        return self.validation_config
    
    def set_info_tabla(self, info_columns):
        """
        Set auxiliary column metadata or info.

        Parameters
        ----------
        info_columns : dict
            Dictionary of metadata or display columns.
        """
        self.info_columns = info_columns
    
    def get_info_tabla(self): 
        """
        Retrieve stored column metadata or information.

        Returns
        -------
        dict
            Column-related information.
        """
        return self.info_columns


class TemporalValidationController:
    """
    Modular controller for executing temporal validation strategies.

    Allows switching between holdout, expanding, and sliding validation modes dynamically.

    Parameters
    ----------
    mode : str, optional
        The validation strategy to use. Must be one of {"holdout", "expanding", "sliding"}.
        Default is "holdout".

    **params : dict
        Additional parameters passed to the selected validation function.

    Attributes
    ----------
    mode : str
        Current validation strategy name.

    params : dict
        Parameters to use for the validation split.

    Methods
    -------
    split(X, y=None)
        Run the configured validation strategy on the input data.
    """
    def __init__(self, mode="holdout", **params):
        """
        Initialize the controller with a selected validation strategy and parameters.

        Parameters
        ----------
        mode : str, optional
            The validation strategy to use. Must be one of {"holdout", "expanding", "sliding"}.
            Default is "holdout".

        **params : dict
            Additional arguments passed to the corresponding split function.

        Attributes
        ----------
        mode : str
            Stores the selected validation strategy.

        params : dict
            Stores the keyword arguments for the chosen strategy.

        _strategy_map : dict
            Internal dictionary mapping strategy names to their corresponding functions.
        """
        self.mode = mode
        self.params = params
        self._strategy_map = {
            "holdout": holdout_split,
            "expanding": expanding_split,
            "sliding": sliding_split,
        }

    def split(self, X, y=None):
        """
        Execute the selected validation strategy on given data.

        Parameters
        ----------
        X : array-like
            Input features or time series data.
        y : array-like, optional
            Target values corresponding to X.

        Returns
        -------
        tuple
            Depends on the selected strategy:
            - "holdout": ((X_train, y_train), (X_val, y_val), (X_test, y_test))
            - "expanding" or "sliding": (folds, (X_test, y_test))
        """
        if self.mode not in self._strategy_map:
            raise ValueError(f"Validation mode '{self.mode}' not supported.")

        split_fn = self._strategy_map[self.mode]
        return split_fn(X, y, **self.params)