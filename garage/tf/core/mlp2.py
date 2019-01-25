"""MLP model in TensorFlow using tf.keras.models.Model."""
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model


def mlp2(input_var,
         output_dim,
         hidden_sizes,
         scope='mlp',
         hidden_nonlinearity='relu',
         hidden_w_init='glorot_uniform',
         hidden_b_init='zeros',
         output_nonlinearity=None,
         output_w_init='glorot_uniform',
         output_b_init='zero',
         batch_normalization=False,
         *args,
         **kwargs):
    """
    MLP model.

    Args:
        input_var: Input tf.Tensor to the MLP.
        output_dim: Dimension of the network output.
        hidden_sizes: Output dimension of dense layer(s).
        scope: variable scope of the mlp.
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).
        batch_normalization: Bool for using batch normalization or not.

    Return:
        The MLP Model.
    """
    with tf.variable_scope(scope):
        _out = input_var
        if isinstance(hidden_sizes, int):
            _out = Dense(
                units=hidden_sizes,
                activation=hidden_nonlinearity,
                kernel_initializer=hidden_w_init,
                bias_initializer=hidden_b_init)(_out)
            if batch_normalization:
                _out = BatchNormalization(
                    hidden_sizes, activation=hidden_nonlinearity)(_out)
        else:
            _out = Dense(
                units=hidden_sizes[0],
                activation=hidden_nonlinearity,
                kernel_initializer=hidden_w_init,
                bias_initializer=hidden_b_init)(_out)
            if batch_normalization:
                _out = BatchNormalization(
                    hidden_sizes[0], activation=hidden_nonlinearity)(_out)
            for hidden_size in hidden_sizes[1:]:
                _out = Dense(
                    units=hidden_size,
                    activation=hidden_nonlinearity,
                    kernel_initializer=hidden_w_init,
                    bias_initializer=hidden_b_init)(_out)
                if batch_normalization:
                    _out = BatchNormalization(
                        hidden_size, activation=hidden_nonlinearity)(_out)
        _out = Dense(
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init)(_out)
        return Model(inputs=input_var, outputs=_out)
