import tensorflow as tf

from garage.tf.models import Model


class SimpleCNNModel(Model):
    """Simple CNNModel for testing."""

    def __init__(self,
                 num_filters,
                 filter_dims,
                 strides,
                 padding,
                 name=None,
                 *args,
                 **kwargs):
        super().__init__(name)
        self.num_filters = num_filters
        self.filter_dims = filter_dims
        self.strides = strides
        self.padding = padding

    def _build(self, obs_input, name=None):
        current_size = obs_input.get_shape().as_list()[1]
        for filter_dim, stride in zip(self.filter_dims, self.strides):
            if self.padding == 'SAME':
                padded = int(filter_dim / 2) * 2
                current_size = int(
                    (current_size - filter_dim + padded) / stride) + 1
            else:
                current_size = int((current_size - filter_dim) / stride) + 1
        flatten_shape = current_size * current_size * self.num_filters[-1]
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        return tf.fill((tf.shape(obs_input)[0], flatten_shape), return_var)
