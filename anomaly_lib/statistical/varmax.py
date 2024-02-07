from statsmodels.tsa.statespace import varmax

import numpy as np

from .. import utils
from .. import algorithmbase

class VARMAX(algorithmbase.AlgorithmBase):

    def __init__(self, hyperparameter: dict, x_train: np.array=None, y_train: np.array = None, x_test: np.array = None,
                 y_test: np.array = None):
        """
            Constructor de la clase VARMAX.

            Args:
                - hyperparametros (dict): Un diccionario que contiene los hiperparámetros para el algoritmo VARMAX. 
                - x_train (np.array): Características de los datos de entrenamiento. La implementación de varmax puede recibir el endog por los hiperparametros si es así no es necesario pasar este parámetro.
                - x_test (np.array): Características de los datos de prueba.
                - y_train (np.array): Etiquetas de los datos de entrenamiento.
                - y_test (np.array): Etiquetas de los datos de prueba.

            Returns:
                None
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
            Ajusta el modelo VARMAX utilizando los datos de entrenamiento.

            Returns:
                Residuals
        """
        self.results = self._model.fit()
        
        return self.results.resid

    def predict(self):
        """
            Ajusta el modelo VARMAX utilizando los datos de entrenamiento.

            Returns:
                None
        """
        x_test = super().get_x_test()

        pred = self.results.forecast(steps=len(x_test))
        

        return super().get_probabilidad_anom(None, x_test, pred)

    def save_model(self, path_model):
        """
            Guarda el modelo VARMAX en el archivo especificado.

            Args:
                - path_model (str): Ruta del archivo donde se guardará el modelo.

            Returns:
                path_model (str): La ruta del archivo donde se guardó el modelo.
        """
        super().save_model(path_model)

    def load_model(self, path_model):
        """
            Carga el modelo VARMAX desde el archivo especificado.

            Args:
                - path_model (str): Ruta del archivo donde se encuentra el modelo.

            Returns:
                model: El modelo VARMAX cargado desde el archivo.
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
        self._model = varmax.VARMAX
        parameters_management = utils.ParameterManagement(self._model.__init__)
        hyperparameter = parameters_management.check_hyperparameter_type(hyperparameter)
        self._hyperparameter = parameters_management.complete_parameters(hyperparameter)

    def _create_model(self, x: np.array = None, y: np.array = None):
        """
            Crea el modelo VARMAX con los hiperparámetros proporcionados.
        """
        self.set_hyperparameter(self._hyperparameter)

        self._model = varmax.VARMAX(**self._hyperparameter)


        