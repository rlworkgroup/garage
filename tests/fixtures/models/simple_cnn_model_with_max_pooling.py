"""Simple CNNModel with max pooling for testing."""
import tensorflow as tf

from garage.tf.models import Model


class SimpleCNNModelWithMaxPooling(Model):
    """Simple CNNModel with max pooling for testing.

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
                 padding,
                 pool_strides,
                 pool_shapes,
                 name=None,
                 hidden_nonlinearity=None,
                 hidden_w_init=None,
                 hidden_b_init=None):
        del hidden_nonlinearity, hidden_w_init, hidden_b_init
        super().__init__(name)
        self.filters = filters
        self.strides = strides
        self.padding = padding
        self.pool_strides = pool_strides
        self.pool_shapes = pool_shapes

    # pylint: disable=arguments-differ
    def _build(self, obs_input, name=None):
        """Build model given input placeholder(s).

        Args:
            obs_input (tf.Tensor): Tensor input for state.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Return:
            tf.Tensor: Tensor output of the model.

        """
        del name
        height_size = obs_input.get_shape().as_list()[1]
        width_size = obs_input.get_shape().as_list()[2]
        for filter_iter, stride in zip(self.filters, self.strides):
            if self.padding == 'SAME':
                height_size = int((height_size + stride - 1) / stride)
                width_size = int((width_size + stride - 1) / stride)
                new_height = height_size + self.pool_strides[0] - 1
                height_size = int(new_height / self.pool_strides[0])
                new_width = width_size + self.pool_strides[1] - 1
                width_size = int(new_width / self.pool_strides[1])
            else:
                height_size = int(
                    (height_size - filter_iter[1][0]) / stride) + 1
                width_size = int((width_size - filter_iter[1][1]) / stride) + 1
                new_height = height_size - self.pool_shapes[0]
                height_size = int(new_height / self.pool_strides[0]) + 1
                new_width = width_size - self.pool_shapes[0]
                width_size = int(new_width / self.pool_strides[1]) + 1
        flatten_shape = height_size * width_size * self.filters[-1][0]
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        return tf.fill((tf.shape(obs_input)[0], flatten_shape), return_var)
