from pyod.models import mo_gaal # mo_gaal.MO_GAAL
import numpy as np

from .. import algorithmbase
from .. import utils


class MO_GAAL(algorithmbase.AlgorithmBase):

    def __init__(self, hyperparameter: dict, x_train: np.array, y_train: np.array = None, x_test: np.array = None,
                 y_test: np.array = None):
        """
            MO_GAAL class constructor.

            Args:
                - hyperparametros (dict): A dictionary containing the hyperparameters for the MO_GAAL algorithm.
                - x_train (np.array): Characteristics of the training data.
                - x_test (np.array): Characteristics of the test data.
                - y_train (np.array): Training data labels.
                - y_test (np.array): Labels of the test data.
        """
        super().__init__(x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test,
                         hyperparameter=hyperparameter)
        
        self._create_model()

    def fit(self):
        """
            It fits the MO_GAAL model using the training data.

            Returns:
                 The outlier scores of the training data [0,1].
        """
        self._model.fit(self._x_train)

        return self._model.decision_scores_

    def predict(self):
        """
            Predict whether a particular sample is an outlier or not.
            Adjusts the MO_GAAL model using the test data.

            Returns:
                None
        """
        x_test = super().get_x_test()

        return self._model.predict_proba(x_test)

    def save_model(self, path_model):
        """
            Guarda el modelo MO_GAAL en el archivo especificado.

            Args:
                - path_model (str): Ruta del archivo donde se guardará el modelo.

            Returns:
                path_model (str): La ruta del archivo donde se guardó el modelo.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
            Carga el modelo MO_GAAL desde el archivo especificado.

            Args:
                - path_model (str): Ruta del archivo donde se encuentra el modelo.

            Returns:
                model: El modelo MO_GAAL cargado desde el archivo.
        """
        super().load_model(path_model)

    def set_hyperparameter(self, hyperparameter):
        """
            Establece los hiperparámetros del modelo.

            Args:
                - hyperparameter (dict): Un diccionario que contiene los hiperparámetros del modelo.

            Returns:
                None
        """
        self._hyperparameter = hyperparameter
        self._model = mo_gaal.MO_GAAL
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
            Crea el modelo MO_GAAL con los hiperparámetros proporcionados.
        """
        self.set_hyperparameter(self._hyperparameter)
        self._model = mo_gaal.MO_GAAL(**self._hyperparameter)