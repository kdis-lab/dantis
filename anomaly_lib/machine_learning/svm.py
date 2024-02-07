from sklearn.svm import SVC
import numpy as np
from .. import algorithmbase
from .. import utils

class SVMClassOD(algorithmbase.AlgorithmBase):

    def __init__(self, hyperparameter: dict, x_train: np.array, y_train: np.array, x_test: np.array = None,
                 y_test: np.array = None):
        """
            SVM class constructor.

            Args:
                - hyperparameters (dict): a dictionary containing the hyperparameters
                  for the SVM algorithm.
                - x_train (np.array): Characteristics of the training data.
                - x_test (np.array): Characteristics of the test data.
                - y_train (np.array): Training data labels.
                - y_test (np.array): Tags of the test data.
        """
        if x_test is not None and y_test is None:
            raise utils.SupervisedInputDataError("X_test is not none and y_test is None")
    
        super().__init__(x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test,
                         hyperparameter=hyperparameter)

        self._create_model()        

    def fit(self):
        """
            Fit detector. Adjusts the SVM model using the training data.

            Returns:
                 The outlier scores of the training data [0,1].

        """
        x_train = self.get_x_train()
        y_train = self.get_y_train()
        self._model.fit(x_train, y_train)
        
        proba_anormalidad = super().get_probabilidad_anom(x_train, y_train)

        return proba_anormalidad

    def predict(self):
        """
            Predict whether a particular sample is an outlier or not.
            Adjusts the SVM model using the test data.

            Returns:
                Predict probability of outlier
        """
        x_test = super().get_x_test()
        y_test = super().get_y_test()
        proba_anormalidad = super().get_probabilidad_anom(x_test, y_test)
        proba_normalidad = 1-proba_anormalidad

        resultado = np.column_stack((proba_normalidad, proba_anormalidad))

        return resultado

    def save_model(self, path_model):
        """
            Saves the SVM model to the specified file.

            Args:
                - path_model (str): Path of the file where the model will be saved.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
            Loads the SVM model from the specified file.

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
        self._model = SVC
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
            Creates the SVM model with the provided hyperparameters.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = SVC(**self._hyperparameter)