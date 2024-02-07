from pyod.models import xgbod # xgbod.XGBOD
import numpy as np

from .. import algorithmbase
from .. import utils


class XGBOD(algorithmbase.AlgorithmBase):

    def __init__(self, hyperparameter: dict, x_train: np.array, y_train: np.array = None, x_test: np.array = None,
                 y_test: np.array = None):
        """
            Constructor de la clase XGBOD.

            Args:
                - hyperparametros (dict): Un diccionario que contiene los hiperparámetros para el algoritmo XGBOD.
                - x_train (np.array): Características de los datos de entrenamiento.
                - x_test (np.array): Características de los datos de prueba.
                - y_train (np.array): Etiquetas de los datos de entrenamiento.
                - y_test (np.array): Etiquetas de los datos de prueba.

            Returns:
                None
        """
        super().__init__(x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test,
                         hyperparameter=hyperparameter)
        
        self._create_model()

    def fit(self):
        """
            Ajusta el modelo XGBOD utilizando los datos de entrenamiento.

            Returns:
                None
        """
        x_train = self.get_x_train()
        y_train = self.get_y_train()
        self._model.fit(self._x_train, y_train)

        return self._model.decision_scores_

    def predict(self):
        """
            Ajusta el modelo XGBOD utilizando los datos de entrenamiento.

            Returns:
                None
        """
        x_test = super().get_x_test()
        return self._model.predict_proba(X=x_test)

    def save_model(self, path_model):
        """
            Guarda el modelo XGBOD en el archivo especificado.

            Args:
                - path_model (str): Ruta del archivo donde se guardará el modelo.

            Returns:
                path_model (str): La ruta del archivo donde se guardó el modelo.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
            Carga el modelo XGBOD desde el archivo especificado.

            Args:
                - path_model (str): Ruta del archivo donde se encuentra el modelo.

            Returns:
                model: El modelo XGBOD cargado desde el archivo.
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
        self._model = xgbod.XGBOD
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self):
        """
            Crea el modelo XGBOD con los hiperparámetros proporcionados.
        """
        self.set_hyperparameter(self._hyperparameter)

        self._model = xgbod.XGBOD(**self._hyperparameter)