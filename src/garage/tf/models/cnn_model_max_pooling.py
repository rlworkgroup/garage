"""CNN Model."""
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models.cnn import cnn_with_max_pooling
from garage.tf.models.model import Model


class CNNModelWithMaxPooling(Model):
    """CNN Model with max pooling.

    Args:
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
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
                 filters,
                 strides,
                 name=None,
                 padding='SAME',
                 pool_strides=(2, 2),
                 pool_shapes=(2, 2),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer()):
        super().__init__(name)
        self._filters = filters
        self._strides = strides
        self._padding = padding
        self._pool_strides = pool_strides
        self._pool_shapes = pool_shapes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init

    # pylint: disable=arguments-differ
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
        del name
        return cnn_with_max_pooling(
            input_var=state_input,
            filters=self._filters,
            hidden_nonlinearity=self._hidden_nonlinearity,
            hidden_w_init=self._hidden_w_init,
            hidden_b_init=self._hidden_b_init,
            strides=self._strides,
            padding=self._padding,
            pool_shapes=self._pool_shapes,
            pool_strides=self._pool_strides,
            name='cnn')
