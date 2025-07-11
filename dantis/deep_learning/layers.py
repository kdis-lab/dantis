
from .. import utils
import tensorflow as tf
from enum import Enum
import tensorflow as tf

class Layer:
    """
    Wrapper class for deep learning layers, managing initialization and hyperparameters.
    This class provides a unified interface to handle different types of layers and their
    associated hyperparameters. It validates input parameters, manages hyperparameter
    completion, and instantiates the appropriate layer object.
    :param layer_type: Enum or object containing layer class and optional wrapper.
    :type layer_type: Any
    :param user_hyperparameters: Dictionary of user-specified hyperparameters for the layer.
    :type user_hyperparameters: dict
    :raises ValueError: If input parameters are invalid.
    Attributes
    ----------
    _layer : Any
        The instantiated layer object.
    _hyperparameter : dict
        Dictionary of completed hyperparameters for the layer.
    Methods
    -------
    check_layers(tuples_layer_type)
        Validates that the provided layer types are not None.
    check_hyperparameters(user_hyperparameters)
        Validates the format and presence of user hyperparameters.
    check_input_parameters(tuples_layer_type, user_hyperparameters)
        Validates both layer types and hyperparameters.
    set_hyperparameter(user_hyperparameters)
        Completes and sets the hyperparameters, and instantiates the layer.
    get_layer()
        Returns the instantiated layer object.
    layer
        Property that returns the instantiated layer object.
    """
    _layer = None
    _hyperparameter = None

    def __init__(self, layer_type, user_hyperparameters):
        """
        Initializes the layer with the specified type and user-defined hyperparameters.
        Parameters
        ----------
        layer_type : Enum
            An enumeration value representing the type of layer to initialize. The value should be a tuple where the first element is the layer class or configuration, and the optional second element is a wrapper or modifier function/class.
        user_hyperparameters : dict
            A dictionary containing user-specified hyperparameters for configuring the layer.
        Raises
        ------
        ValueError
            If the input parameters are invalid as determined by `check_input_parameters`.
        """
        self.check_input_parameters(layer_type.value, user_hyperparameters)
        self._layer = layer_type.value[0]

        self.set_hyperparameter(user_hyperparameters)

        if len(layer_type.value) > 1:
            self._layer = layer_type.value[1](self._layer)
        

    @staticmethod
    def check_layers(tuples_layer_type):
        """
        Validates that all elements in the input iterable are not None.
        Parameters
        ----------
        tuples_layer_type: Iterable
            An iterable containing layer types to be checked.
        Raises
        ------
        ValueError
            If any element in tuples_layer_type is None.
        """
        for layer_type in tuples_layer_type:
            if layer_type is None:
                raise ValueError("Error, layer_type is None")

    @staticmethod
    def check_hyperparameters(user_hyperparameters):
        """
        Validates the user-provided hyperparameters.
        Checks if the input is not None and is a dictionary. Raises a ValueError if the checks fail.

        Parameters
        ----------
        user_hyperparameters: Dict or None
            The hyperparameters provided by the user.
        Raises
        ------
        ValueError
            If user_hyperparameters is None or not a dictionary.
        """

        if user_hyperparameters is None:
            raise ValueError("Error, user hyperparameter is None")

        if not isinstance(user_hyperparameters, dict):
            raise ValueError("Error format, hyperparameter must be a dict")

    def check_input_parameters(self, tuples_layer_type, user_hyperparameters):
        self.check_layers(tuples_layer_type)
        self.check_hyperparameters(user_hyperparameters)

    def set_hyperparameter(self, user_hyperparameters):
        """
        Sets and completes the hyperparameters for the layer.
        If the layer is a custom transformer block, initializes it with specific parameters
        (num_heads, ff_dim, dropout) from user_hyperparameters or their default values.
        Otherwise, completes the hyperparameters using ParameterManagement and initializes
        the layer with them.
        Parameters
        ----------
        user_hyperparameters : dict
            Dictionary containing the hyperparameters provided by the user.
        Raises
        ------
        ValueError
            If user_hyperparameters is None or not a dictionary.
        """
        hpm = utils.ParameterManagement(self._layer.__init__)
        self._hyperparameter = hpm.complete_parameters(user_hyperparameters)

        if self._layer == "CUSTOM_TRANSFORMER_BLOCK":
            num_heads = user_hyperparameters.get("num_heads", 4)
            ff_dim = user_hyperparameters.get("ff_dim", 64)
            dropout = user_hyperparameters.get("dropout", 0.1)
            self._layer = TransformerBlock(num_heads=num_heads, ff_dim=ff_dim, dropout=dropout)
        else:
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
    TRANSFORMER_BLOCK = ("CUSTOM_TRANSFORMER_BLOCK",)




class TransformerBlock(tf.keras.layers.Layer):
    """
    TransformerBlock implements a single transformer encoder block for multivariate time series data.
    This layer applies multi-head self-attention followed by a feed-forward neural network, each with residual connections and layer normalization. It is compatible with input shapes of (batch_size, seq_len, n_feat).
    Parameters
    ----------
    num_heads : int
        Number of attention heads in the MultiHeadAttention layer.
    ff_dim : int
        Dimensionality of the feed-forward network and the key dimension in attention.
    dropout : float, optional
        Dropout rate applied after attention and feed-forward layers (default is 0.1).
    Methods
    -------
    call(inputs, training=False)
        Applies the transformer block to the input tensor.
    Returns
    -------
    tf.Tensor
        Output tensor of the same shape as the input.
    Raises
    ------
    ValueError
        If input tensor shape is incompatible.
    """
    
    def __init__(self, num_heads: int, ff_dim: int, dropout: float = 0.1):
        """
        Initializes the TransformerBlock with the specified number of attention heads, feed-forward dimension, and dropout rate.
        Parameters
        ----------
        num_heads : int
            Number of attention heads in the MultiHeadAttention layer.
        ff_dim : int
            Dimensionality of the feed-forward network and the key dimension in attention.  
        dropout : float, optional
            Dropout rate applied after attention and feed-forward layers (default is 0.1).  
        """
        super(TransformerBlock, self).__init__()
        self.att = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=ff_dim)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="relu"),
            tf.keras.layers.Dense(ff_dim)
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout)
        self.dropout2 = tf.keras.layers.Dropout(dropout)

    def call(self, inputs, training=False):
        """
        Applies the transformer block to the input tensor.
        Parameters
        ----------
        inputs : tf.Tensor
            Input tensor of shape (batch_size, seq_len, n_feat).
        training : bool, optional
            Whether the layer should behave in training mode (default is False).
                
        Returns
        -------
        tf.Tensor
            Output tensor of the same shape as the input.
        """
        attn_output = self.att(inputs, inputs, attention_mask=None)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
