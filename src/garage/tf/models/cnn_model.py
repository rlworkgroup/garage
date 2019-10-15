"""CNN Model."""
import tensorflow as tf

from garage.tf.models.base import Model
from garage.tf.models.cnn import cnn


class CNNModel(Model):
    """CNN Model.

    Args:
        filter_dims(tuple[int]): Dimension of the filters. For example,
            (3, 5) means there are two convolutional layers. The filter
            for first layer is of dimension (3 x 3) and the second one is of
            dimension (5 x 5).
        num_filters(tuple[int]): Number of filters. For example, (3, 32) means
            there are two convolutional layers. The filter for the first layer
            has 3 channels and the second one with 32 channels.
        strides(tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        name (str): Model name, also the variable scope.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
    """

    def __init__(self,
                 filter_dims,
                 num_filters,
                 strides,
                 padding,
                 name=None,
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer()):
        super().__init__(name)
        self._filter_dims = filter_dims
        self._num_filters = num_filters
        self._strides = strides
        self._padding = padding
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init

    def _build(self, state_input, name=None):
        return cnn(input_var=state_input,
                   filter_dims=self._filter_dims,
                   hidden_nonlinearity=self._hidden_nonlinearity,
                   hidden_w_init=self._hidden_w_init,
                   hidden_b_init=self._hidden_b_init,
                   num_filters=self._num_filters,
                   strides=self._strides,
                   padding=self._padding,
                   name='cnn')
