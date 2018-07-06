"""Spaces.Discrete for TensorFlow."""
import numpy as np
import tensorflow as tf

from garage.spaces import Discrete as GarageDiscrete


class Discrete(GarageDiscrete):
    """TensorFlow extension of garage.Discrete."""

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable in TensorFlow.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :return: the created tensor variable
        """
        # needed for safe conversion to float32
        return tf.placeholder(
            dtype=tf.uint8,
            shape=[None] * extra_dims + [self.flat_dim],
            name=name)

    @property
    def dtype(self):
        """
        Return the Tensor element's type.

        :return: data type of the Tensor element
        """
        return tf.uint8

    def sample_n(self, n):
        """
        Return an ndarray of random integers from 0 to n-1 inclusive.

        :param n: size of output
        :return: ndarray of ints
        """
        return np.random.randint(low=0, high=self.n, size=n)
