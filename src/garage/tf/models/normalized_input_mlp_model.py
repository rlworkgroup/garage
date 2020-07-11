"""NormalizedInputMLPModel."""
import numpy as np
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models.mlp_model import MLPModel


class NormalizedInputMLPModel(MLPModel):
    """NormalizedInputMLPModel based on garage.tf.models.Model class.

    This class normalized the inputs and pass the normalized input to a
    MLP model, which can be used to perform linear regression to the outputs.

    Args:
        input_shape (tuple[int]): Input shape of the training data.
        output_dim (int): Output dimension of the model.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='NormalizedInputMLPModel',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        super().__init__(output_dim=output_dim,
                         name=name,
                         hidden_sizes=hidden_sizes,
                         hidden_nonlinearity=hidden_nonlinearity,
                         hidden_w_init=hidden_w_init,
                         hidden_b_init=hidden_b_init,
                         output_nonlinearity=output_nonlinearity,
                         output_w_init=output_w_init,
                         output_b_init=output_b_init,
                         layer_normalization=layer_normalization)
        self._input_shape = input_shape

    def network_output_spec(self):
        """Network output spec.

        Return:
            list[str]: List of key(str) for the network outputs.

        """
        return ['y_hat', 'x_mean', 'x_std']

    def _build(self, state_input, name=None):
        """Build model given input placeholder(s).

        Args:
            state_input (tf.Tensor): Tensor input for state.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Return:
            tf.Tensor: Tensor output of the model.

        """
        with tf.compat.v1.variable_scope('normalized_vars'):
            x_mean_var = tf.compat.v1.get_variable(
                name='x_mean',
                shape=(1, ) + self._input_shape,
                dtype=np.float32,
                initializer=tf.zeros_initializer(),
                trainable=False)
            x_std_var = tf.compat.v1.get_variable(
                name='x_std_var',
                shape=(1, ) + self._input_shape,
                dtype=np.float32,
                initializer=tf.ones_initializer(),
                trainable=False)

        normalized_xs_var = (state_input - x_mean_var) / x_std_var

        y_hat = super()._build(normalized_xs_var)

        return y_hat, x_mean_var, x_std_var
