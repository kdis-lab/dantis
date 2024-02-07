from pyod.models import vae # vae.VAE
import numpy as np
import joblib
import keras

from .. import algorithmbase
from .. import utils


class VAE(algorithmbase.AlgorithmBase):

    def __init__(self, hyperparameter: dict, x_train: np.array, y_train: np.array = None, x_test: np.array = None,
                 y_test: np.array = None):
        """
            VAE class constructor.

            Args:
                - hyperparametros (dict): A dictionary containing the hyperparameters for the VAE algorithm.
                - x_train (np.array): Characteristics of the training data.
                - x_test (np.array): Characteristics of the test data.
                - y_train (np.array): Training data labels.
                - y_test (np.array): Labels of the test data.
        """
        super().__init__(x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test,
                         hyperparameter=hyperparameter)
        
        self._create_model()
        print(self._x_test.shape)

    def fit(self):
        """
            Fit detector. Adjusts the VAE model using the training data.

            Returns:
                 The outlier scores of the training data [0,1].
        """
        self._model.fit(self._x_train)

        return self._model.decision_scores_

    def predict(self):
        """
            Predict whether a particular sample is an outlier or not.
            Adjusts the VAE model using the test data.

            Returns:
                Predict probability of outlier
        """
        x_test = super().get_x_test()

        return self._model.predict_proba(x_test)

    def save_model(self, path_model):
        """
            Saves the VAE model to the specified file.

            Args:
                - path_model (str): Path of the file where the model will be saved.
        """
        extension = "extension"
        if "." in path_model:
            extension = path_model[path_model.rfind(".")+1:]
            if extension != "joblib":
                print("Warning: Change extension")
                extension = "joblib"
        self._model.model_.save("model")
        self._model.model_ = None
        path_model = path_model[:path_model.rfind(".")+1] + extension
        joblib.dump(self._model, path_model)

    def load_model(self, path_model):
        """
            Loads the VAE model from the specified file.

            Args:
                - path_model (str): Path of the file where the model is located.
        """
        if "." in path_model:
            extension = path_model[path_model.rfind(".")+1:]
            if extension != "joblib":
                raise (Exception("Error: extension required .joblib"))

        self._model = joblib.load(path_model)
        self._model.model_ = keras.models.load_model("model", custom_objects={'sampling': self._model.sampling})

    def set_hyperparameter(self, hyperparameter):
        """
            Sets the hyperparameters of the model.

            Args:
                - hyperparameter (dict): A dictionary containing the hyperparameters of the model.
        """
        self._hyperparameter = hyperparameter
        self._model = vae.VAE
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
            Creates the VAE model with the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = vae.VAE(**self._hyperparameter)