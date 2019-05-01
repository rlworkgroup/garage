"""CNN Model."""
import tensorflow as tf

from garage.tf.core.cnn import cnn_with_max_pooling
from garage.tf.models.base import Model


class CNNModelWithMaxPooling(Model):
    """
    CNN Model with max pooling.

    Args:
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        strides: The stride of the sliding window.
        name: Variable scope of the cnn.
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        pool_shapes: Dimension of the pooling layer(s).
        pool_strides: The stride of the pooling layer(s).
        name: Variable scope of the cnn.
        hidden_nonlinearity: Activation function for intermediate dense
            layer(s).
    """

    def __init__(self,
                 filter_dims,
                 num_filters,
                 strides,
                 name=None,
                 padding='SAME',
                 pool_strides=(2, 2),
                 pool_shapes=(2, 2),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer()):
        super().__init__(name)
        self._filter_dims = filter_dims
        self._num_filters = num_filters
        self._strides = strides
        self._padding = padding
        self._pool_strides = pool_strides
        self._pool_shapes = pool_shapes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init

    def _build(self, state_input, name=None):
        return cnn_with_max_pooling(
            input_var=state_input,
            filter_dims=self._filter_dims,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            num_filters=self._num_filters,
            strides=self._strides,
            padding=self._padding,
            pool_shapes=self._pool_shapes,
            pool_strides=self._pool_strides,
            name='cnn')
