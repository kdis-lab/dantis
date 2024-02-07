
from .. import utils
import tensorflow as tf
from enum import Enum


class Layer:
    _layer = None
    _hyperparameter = None

    def __init__(self, layer_type, user_hyperparameters):
        self.check_input_parameters(layer_type.value, user_hyperparameters)
        self._layer = layer_type.value[0]

        self.set_hyperparameter(user_hyperparameters)

        if len(layer_type.value) > 1:
            self._layer = layer_type.value[1](self._layer)
        

    @staticmethod
    def check_layers(tuples_layer_type):
        for layer_type in tuples_layer_type:
            if layer_type is None:
                raise ValueError("Error, layer_type is None")

    @staticmethod
    def check_hyperparameters(user_hyperparameters):
        if user_hyperparameters is None:
            raise ValueError("Error, user hyperparameter is None")

        if not isinstance(user_hyperparameters, dict):
            raise ValueError("Error format, hyperparameter must be a dict")

    def check_input_parameters(self, tuples_layer_type, user_hyperparameters):
        self.check_layers(tuples_layer_type)
        self.check_hyperparameters(user_hyperparameters)

    def set_hyperparameter(self, user_hyperparameters):
        hpm = utils.ParameterManagement(self._layer.__init__)
        self._hyperparameter = hpm.complete_parameters(user_hyperparameters)

        self._layer = self._layer(**self._hyperparameter)

    def get_layer(self):
        return self._layer

    @property
    def layer(self):
        return self._layer


class LayerType(Enum):
    LSTM = (tf.keras.layers.LSTM,)
    CNN = (tf.keras.layers.Conv1D,)
    GRU = (tf.keras.layers.GRU,)
    BIDIRECTIONAL_LSTM = (tf.keras.layers.LSTM, tf.keras.layers.Bidirectional)
    BIDIRECTIONAL_GRU = (tf.keras.layers.GRU, tf.keras.layers.Bidirectional)
    DENSE = (tf.keras.layers.Dense,)
    DROPOUT = (tf.keras.layers.Dropout,)
    TIMEDISTRIBUTED = (tf.keras.layers.TimeDistributed,)
    CNNLSTM1D = (tf.keras.layers.ConvLSTM1D,)
    CNNLSTM2D = (tf.keras.layers.ConvLSTM2D,)
    CNNLSTM3D = (tf.keras.layers.ConvLSTM3D,)
    MAXPOOLING_1D = (tf.keras.layers.MaxPooling1D,)
    UPSAMPLING_1D = (tf.keras.layers.UpSampling1D,)
    MAXPOOLING_2D = (tf.keras.layers.MaxPooling2D,)
    UPSAMPLING_2D = (tf.keras.layers.UpSampling2D,)   
    MAXPOOLING_3D = (tf.keras.layers.MaxPooling3D,)
    UPSAMPLING_3D = (tf.keras.layers.UpSampling3D,)

