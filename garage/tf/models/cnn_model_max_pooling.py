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
        pool_shape: Dimension of the pooling layer(s).
        pool_stride: The stride of the pooling layer(s).
        name: Variable scope of the cnn.
        hidden_nonlinearity: Activation function for intermediate dense
            layer(s).
    """

    def __init__(self,
                 filter_dims,
                 num_filters,
                 strides,
                 name=None,
                 padding="SAME",
                 pool_stride=(2, 2),
                 pool_shape=(2, 2),
                 hidden_nonlinearity=tf.nn.relu):
        super().__init__(name)
        self._filter_dims = filter_dims
        self._num_filters = num_filters
        self._strides = strides
        self._padding = padding
        self._pool_stride = pool_stride
        self._pool_shape = pool_shape
        self._hidden_nonlinearity = hidden_nonlinearity

    def _build(self, state_input):
        return cnn_with_max_pooling(
            input_var=state_input,
            filter_dims=self._filter_dims,
            hidden_nonlinearity=self._hidden_nonlinearity,
            num_filters=self._num_filters,
            strides=self._strides,
            padding=self._padding,
            pool_shape=self._pool_shape,
            pool_stride=self._pool_stride,
            name="cnn")
