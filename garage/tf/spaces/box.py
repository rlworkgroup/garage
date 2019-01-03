"""Spaces.Box for TensorFlow."""
import tensorflow as tf

from garage.spaces import Box as GarageBox


class Box(GarageBox):
    """TensorFlow extension of garage.Box."""

    def new_tensor_variable(self, name, extra_dims, flatten=True):
        """
        Create a tensor variable in TensorFlow.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :param flatten: whether to flatten the shape
        :return: the created tensor variable
        """
        if flatten:
            return tf.placeholder(
                self.dtype,
                shape=[None] * extra_dims + [self.flat_dim],
                name=name)
        return tf.placeholder(
            self.dtype,
            shape=[None] * extra_dims + list(self.shape),
            name=name)
