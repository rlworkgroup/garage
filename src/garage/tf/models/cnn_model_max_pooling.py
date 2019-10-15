"""CNN Model."""
import tensorflow as tf

from garage.tf.models.base import Model
from garage.tf.models.cnn import cnn_with_max_pooling


class CNNModelWithMaxPooling(Model):
    """CNN Model with max pooling.

    Args:
        filter_dims (tuple[int]): Dimension of the filters. For example,
            (3, 5) means there are two convolutional layers. The filter for
            first layer is of dimension (3 x 3) and the second one is of
            dimension (5 x 5).
        num_filters (tuple[int]): Number of filters. For example, (3, 32) means
            there are two convolutional layers. The filter for the first layer
            has 3 channels and the second one with 32 channels.
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        name (str): Model name, also the variable scope of the cnn.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        pool_strides (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        pool_shapes (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
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
