"""CNN Model."""
import tensorflow as tf

from garage.tf.core.cnn import cnn
from garage.tf.models.base import Model


class CNNModel(Model):
    """
    CNN Model.

    Args:
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        strides: The stride of the sliding window.
        name: Variable scope of the cnn.
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        hidden_nonlinearity: Activation function for intermediate dense
            layer(s).
    """

    def __init__(self,
                 filter_dims,
                 num_filters,
                 strides,
                 padding,
                 name=None,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 hidden_nonlinearity=tf.nn.relu):
        super().__init__(name)
        self._filter_dims = filter_dims
        self._num_filters = num_filters
        self._strides = strides
        self._padding = padding
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init

    def _build(self, state_input):
        return cnn(
            input_var=state_input,
            filter_dims=self._filter_dims,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            num_filters=self._num_filters,
            strides=self._strides,
            padding=self._padding,
            name="cnn")
