import tensorflow as tf
from . import layers 
from .. import utils
from .. import algorithmbase
import copy
import numpy as np


class Topology(algorithmbase.AlgorithmBase):
    _model = None




    def __init__(self, hyperparameter: dict, x_train: np.array=None, y_train: np.array = None, x_test: np.array = None,
                 y_test: np.array = None):
        
        """
        Constructor de la clase Topology.

        Args:
            - hyperparametros (dict): Un diccionario que contiene los hiperparámetros
                para el algoritmo Topology. Este recive obligatoriamente los siguientes parametros:

                   hyperparam = {"input_shape": (100, 3),
                    "layers": [l1, l2, l3, l4]],
                    "compile": {"optimizer":"adam", "loss":"mse" ......},
                    "fit": {"x": X_train, "y": X_train, .... }
                }

            - x_train (np.array): Se mantienen por consistencia con los otros modelos, pero no se usan.
            - x_test (np.array): Se mantienen por consistencia con los otros modelos, pero no se usan.
            - y_train (np.array): Se mantienen por consistencia con los otros modelos, pero no se usan.
            - y_test (np.array): Se mantienen por consistencia con los otros modelos, pero no se usan.

        Returns:
            None
        """
        
        super().__init__(x_train=x_train, x_test=x_test,
                         y_train=y_train, y_test=y_test,
                         hyperparameter=hyperparameter)
        
   
        
        input_shape=hyperparameter

        input_shape=tf.keras.Input(shape=(hyperparameter["input_shape"][0], hyperparameter["input_shape"][1]))

        fixed_layer = []

        for id_layer, layer in enumerate(hyperparameter["layers"]):
            layer_conf = self.__get_layer_from_list_(layer) if isinstance(layer, list) else layer
            layer_conf._name += str(id_layer)
            fixed_layer.append(copy.deepcopy(layer_conf))


        
        #print(names)
        self.define_topology(input_shape, fixed_layer)
        self._model = self.compile_topology(hyperparameter["compile"])



    @staticmethod
    def __get_layer_from_list_(layer_hyperparam):
        
        if isinstance(layer_hyperparam, list):
            if len(layer_hyperparam) == 2:
                return layers.Layer(layers.LayerType[layer_hyperparam[0]], user_hyperparameters=layer_hyperparam[1])

            else:
                raise ValueError("Error, layer must recive [Type of layer, \{hyperparam name: [value, type]\}]")
        else:
            raise ValueError("Layer is not a list")




    def define_topology(self, input_shape, layers):
        if hasattr(layers, "__iter__"):
            self._model = tf.keras.Sequential()
            self._model.add(input_shape)
            for layer_to_add in layers:
                self._model.add(layer_to_add)

        else:
            raise ValueError("Layers must be an iterable object")

    def compile_topology(self, compile_params):
        if self._model is None:
            raise ValueError("Error, topology must be defined first")

        self._model.compile(**compile_params)
        return self._model

    def fit(self, parameters_fit=None):
        if parameters_fit is None:
            if self._hyperparameter["fit"] is None:
                raise ValueError("The method did not received any fit parameters and the"
                                 + "class received no fit parameters in the constructor.")
            else:
                parameters_fit = self._hyperparameter["fit"]

        hpm = utils.ParameterManagement(self._model.fit)
        parameters_fit = hpm.complete_parameters(parameters_fit)

        self._model.fit(**parameters_fit)

    def get_topology(self):
        return self._model

    def set_topology(self, model):
        self._model = model

    def predict(self, x):
        return self._model.predict(x)
    

    def predict_reconstruction_error(self, x, y):
        pred = self.predict(x)

        if pred.shape != y.shape:
            raise ValueError("pred.shape:" + str(pred.shape) + "needs to be equal to y.shape:" + str(y.shape))
        
        return np.abs(np.subtract(pred, y)) 
