"""Spaces.Tuple for TensorFlow."""
import tensorflow as tf

from garage.spaces import Tuple as GarageTuple


class Tuple(GarageTuple):
    """TensorFlow extension of garage.Tuple."""

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable in TensorFlow.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :return: the created tensor variable
        """
        return tf.placeholder(
            dtype=self._common_dtype,
            shape=[None] * extra_dims + [self.flat_dim],
            name=name,
        )

    @property
    def dtype(self):
        """
        Return the Tensor element's type.

        :return: data type of the Tensor element
        """
        return self._common_dtype
