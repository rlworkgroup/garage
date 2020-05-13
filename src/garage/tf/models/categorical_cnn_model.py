"""Categorical CNN Model.

A model represented by a Categorical distribution
which is parameterized by a convolutional neural network (CNN)
followed a multilayer perceptron (MLP).
"""
import tensorflow as tf

from garage.tf.models.categorical_mlp_model import CategoricalMLPModel
from garage.tf.models.cnn_model import CNNModel
from garage.tf.models.model import Model


class CategoricalCNNModel(Model):
    """Categorical CNN Model.

    A model represented by a Categorical distribution
    which is parameterized by a convolutional neural network (CNN) followed
    by a multilayer perceptron (MLP).

    Args:
        output_dim (int): Dimension of the network output.
        filter_dims (tuple[int]): Dimension of the filters. For example,
            (3, 5) means there are two convolutional layers. The filter
            for first layer is of dimension (3 x 3) and the second one is of
            dimension (5 x 5).
        num_filters (tuple[int]): Number of filters. For example, (3, 32) means
            there are two convolutional layers. The filter for the first layer
            has 3 channels and the second one with 32 channels.
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        name (str): Model name, also the variable scope.
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
                 output_dim,
                 filter_dims,
                 num_filters,
                 strides,
                 padding,
                 name=None,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(),
                 output_b_init=tf.zeros_initializer(),
                 layer_normalization=False):
        super().__init__(name)
        self._cnn_model = CNNModel(filter_dims=filter_dims,
                                   num_filters=num_filters,
                                   strides=strides,
                                   padding=padding,
                                   hidden_nonlinearity=hidden_nonlinearity,
                                   name='CNNModel')
        self._mlp_model = CategoricalMLPModel(
            output_dim=output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization,
            name='MLPModel')

    # pylint: disable=arguments-differ
    def _build(self, state_input, name=None):
        """Build model.

        Args:
            state_input (tf.Tensor): Observation inputs.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Returns:
            tfp.distributions.OneHotCategorical: Policy distribution.

        """
        time_dim = tf.shape(state_input)[1]
        dim = state_input.get_shape()[2:].as_list()
        state_input = tf.reshape(state_input, [-1, *dim])
        cnn_output = self._cnn_model.build(state_input, name=name)
        dim = cnn_output.get_shape()[-1]
        cnn_output = tf.reshape(cnn_output, [-1, time_dim, dim])
        mlp_output = self._mlp_model.build(cnn_output, name=name)
        return mlp_output
