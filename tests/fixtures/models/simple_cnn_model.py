"""Simple CNNModel for testing."""
import tensorflow as tf

from garage.tf.models import Model


# pylint: disable=missing-param-doc, missing-type-doc
# pylint: disable=missing-return-type-doc, missing-return-doc
class SimpleCNNModel(Model):
    """Simple CNNModel for testing."""

    # pylint: disable=keyword-arg-before-vararg, unused-argument
    def __init__(self, filters, strides, padding, name=None, *args, **kwargs):
        super().__init__(name)
        self.filters = filters
        self.strides = strides
        self.padding = padding

    # pylint: disable=arguments-differ
    def _build(self, obs_input, name=None):
        height_size = obs_input.get_shape().as_list()[1]
        width_size = obs_input.get_shape().as_list()[2]
        for filter_iter, stride in zip(self.filters, self.strides):
            if self.padding == 'SAME':
                height_size = int((height_size + stride - 1) / stride)
                width_size = int((width_size + stride - 1) / stride)
            else:
                height_size = int(
                    (height_size - filter_iter[0][0]) / stride) + 1
                width_size = int((width_size - filter_iter[0][1]) / stride) + 1
        flatten_shape = height_size * width_size * self.filters[-1][-1]
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        return tf.fill((tf.shape(obs_input)[0], flatten_shape), return_var)
