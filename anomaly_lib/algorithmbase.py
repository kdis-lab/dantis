import numpy as np 
import joblib

class AlgorithmBase:
    # Class Attribute
    _model = None
    _hyperparameter = None
    _x_train = None
    _x_test = None
    _y_train = None
    _y_test = None

    def __init__(self, hyperparameter: dict, x_train: np.array, y_train: np.array = None, x_test: np.array = None,
                 y_test: np.array = None) -> None:
        self._hyperparameter = hyperparameter
        self._x_train = x_train
        self._x_test = x_test
        self._y_train = y_train
        self._y_test = y_test
        
    def set_hyperparameter(self, hyperparameter):
        self._hyperparameter = hyperparameter if not (hyperparameter is None) else None \
            if self._hyperparameter is None else self._hyperparameter

    def get_hyperparameter(self):
        return self._hyperparameter

    def set_x_train(self, x_train):
        self._x_train = x_train if not(x_train is None) else None if self._x_train is None else self._x_train

    def get_x_train(self):
        return self._x_train

    def set_x_test(self, x_test):
        self._x_test = x_test if not(x_test is None) else None if self._x_test is None else self._x_test

    def get_x_test(self):
        return self._x_test

    def set_y_train(self, y_train):
        self._y_train = y_train if not(y_train is None) else None if self._y_train is None else self._y_train

    def get_y_train(self):
        return self._y_train

    def set_y_test(self, y_test):
        self._y_test = y_test if not(y_test is None) else None if self._y_test is None else self._y_test

    def get_y_test(self):
        return self._y_test

    def fit(self):
        pass

    def predict(self):
        pass

    def save_model(self, path_model: str):
        if "." in path_model:
            extension = path_model[path_model.rfind(".")+1:]
            if extension != "joblib":
                print("Warning: Change extension")
                extension = "joblib"

        path_model = path_model[:path_model.rfind(".")+1] + extension
        joblib.dump(self._model, path_model)

    def load_model(self, path_model: str):
        if "." in path_model:
            extension = path_model[path_model.rfind(".")+1:]
            if extension != "joblib":
                raise(Exception("Error: extension required .joblib"))

        self._model = joblib.load(path_model)

    def get_probabilidad_anom(self, X, y=None, pred=None):
        if pred is None:
            pred = self._model.predict(X)  

        if y is None:
            return pred
        
 
        decision_scores = np.abs(pred-y)
        min_error = np.min(decision_scores)
        max_error = np.max(decision_scores)
        if max_error - min_error >= 0.0000000001:
            proba_anormalidad = (decision_scores-min_error) / (max_error-min_error)
        else:
            proba_anormalidad = decision_scores
        return proba_anormalidad