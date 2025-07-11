def holdout_split(X, y, val_size=0.1, test_size=0.2, **kwargs):
    n = len(X)
    test_split = int(n * (1 - test_size))
    val_split = int(test_split * (1 - val_size))

    X_train, X_val, X_test = X[:val_split], X[val_split:test_split], X[test_split:]
    y_train = y_val = y_test = None
    if y is not None:
        y_train, y_val, y_test = y[:val_split], y[val_split:test_split], y[test_split:]
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def expanding_split(X, y, train_size=100, step_size=20, test_size=0.2, **kwargs):
    n = len(X)
    test_split = int(n * (1 - test_size))
    folds = []

    start = 0
    while True:
        end_train = start + train_size
        end_val = end_train + step_size

        if end_val >= test_split:
            break

        X_train, X_val = X[start:end_train], X[end_train:end_val]
        y_train = y_val = None
        if y is not None:
            y_train, y_val = y[start:end_train], y[end_train:end_val]
        
        folds.append(((X_train, y_train), (X_val, y_val)))

        start = 0  # expanding: el train siempre empieza en 0 y crece
        train_size += step_size  # se expande

    X_test = X[test_split:]
    y_test = y[test_split:] if y is not None else None
    return folds, (X_test, y_test)


def sliding_split(X, y, train_size=100, step_size=20, test_size=0.2, **kwargs):
    n = len(X)
    test_split = int(n * (1 - test_size))
    folds = []

    start = 0
    while True:
        end_train = start + train_size
        end_val = end_train + step_size

        if end_val >= test_split:
            break

        X_train, X_val = X[start:end_train], X[end_train:end_val]
        y_train = y_val = None
        if y is not None:
            y_train, y_val = y[start:end_train], y[end_train:end_val]
        
        folds.append(((X_train, y_train), (X_val, y_val)))
        start += step_size  # sliding: se desliza

    X_test = X[test_split:]
    y_test = y[test_split:] if y is not None else None
    return folds, (X_test, y_test)



class ValidationController: 
    def __init__(self):
        self.validation_config = {}
        self.info_columns = {}

    def set_validation_config (self, validation_config): 
        self.validation_config =  validation_config

    def get_validation_config (self): 
        return self.validation_config
    
    def set_info_tabla(self, info_columns):
        self.info_columns = info_columns
    
    def get_info_tabla(self): 
        return self.info_columns


class TemporalValidationController:
    """
    Modular controller for temporal validation strategies.

    Parameters
    ----------
    mode : str
        Validation mode to use. One of {"holdout", "expanding", "sliding"}.
    **params : dict
        Additional parameters passed to the selected validation function.
    """
    def __init__(self, mode="holdout", **params):
        self.mode = mode
        self.params = params
        self._strategy_map = {
            "holdout": holdout_split,
            "expanding": expanding_split,
            "sliding": sliding_split,
        }

    def split(self, X, y=None):
        """
        Execute the selected validation strategy.

        Parameters
        ----------
        X : array-like
            Feature matrix or time series input.
        y : array-like, optional
            Target values or labels.

        Returns
        -------
        tuple
            Depends on strategy:
            - holdout: ((X_train, y_train), (X_val, y_val), (X_test, y_test))
            - expanding/sliding: (folds, (X_test, y_test))
        """
        if self.mode not in self._strategy_map:
            raise ValueError(f"Validation mode '{self.mode}' not supported.")

        split_fn = self._strategy_map[self.mode]
        return split_fn(X, y, **self.params)


if __name__ == "__main__":
    import numpy as np

    X = np.sin(np.linspace(0, 10 * np.pi, 1000)) + np.random.normal(0, 0.1, 1000)
    y = list(range(len(X)))


    val = TemporalValidationController(
        mode="sliding",
        train_size=200,
        step_size=50,
        test_size=0.2
    )
    model = None
    folds, (X_test, y_test) = val.split(X, y)
    for (X_train, y_train), (X_val, y_val) in folds:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

    controller = TemporalValidationController(
        mode="holdout",
        val_size=0.1,
        test_size=0.2
    )
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = controller.split(X, y)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    y_test_pred = model.predict(X_test)