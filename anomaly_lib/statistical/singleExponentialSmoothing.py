
from statsmodels.tsa.exponential_smoothing import ets 
import numpy as np

from .. import utils
from .. import algorithmbase


class SingleExponentialSmoothing(algorithmbase.AlgorithmBase):

    def __init__(self, hyperparameter: dict, x_train: np.array=None, y_train: np.array = None, x_test: np.array = None,
                 y_test: np.array = None):
        """
            Simple Exponential Smoothing class constructor.

            Args:
                - hyperparameters (dict): A dictionary containing the hyperparameters for the Simple Exponential Smoothing algorithm.
                - x_train (np.array): Characteristics of the training data. The Simple Exponential Smoothing implementation may receive the
                  endog for the hyperparameters. if so it is not necessary to pass this parameter.
                - x_test (np.array): Test data characteristics.
                - y_train (np.array): Training data labels.
                - y_test (np.array): Test data labels.
        """
        if "endog" not in hyperparameter.keys() and x_train != None:
            hyperparameter["endog"] = x_train
        
        
        if "endog" in hyperparameter.keys() and x_train != None:
            if hyperparameter["endog"] is None:
                hyperparameter["endog"] = x_train

        if x_train is None:
            x_train = hyperparameter["endog"]

        super().__init__(x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test,
                         hyperparameter=hyperparameter)
        
        self._create_model() 

    def fit(self):
        """
            Fit detector. Adjusts the Simple Exponential Smoothing model using the training data.

            Returns:
                 The outlier scores of the training data.
        """
        self.results = self._model.fit()
        
        return self.results.resid

    def predict(self):
        """
             Predict whether a particular sample is an outlier or not.
             Adjusts the Simple Exponential Smoothing model using the test data.

             Returns:
                 Predict probability of outlier
         """
        x_test = super().get_x_test()

        pred = self.results.forecast(steps=len(x_test))

        return super().get_probabilidad_anom(None, x_test, pred)

    def save_model(self, path_model):
        """
            Saves the Simple Exponential Smoothing model to the specified file.

            Args:
                - path_model (str): Path of the file where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
            Loads the Simple Exponential Smoothing model from the specified file.

            Args:
                - path_model (str): Path of the file where the model is located.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
             Sets the hyperparameters of the model.

             Args:
                 - hyperparameter (dict): A dictionary containing the hyperparameters of the model.
        """
        self._hyperparameter = hyperparameter
        self._model = ets.ETSModel
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
            Creates the Simple Exponential Smoothing model with the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = ets.ETSModel(**self._hyperparameter)