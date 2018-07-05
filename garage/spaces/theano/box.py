"""Spaces.Box for Theano."""
import theano

from garage.misc import ext
from garage.spaces import Box as GarageBox


class Box(GarageBox):
    """Theano extension of garage.Box."""

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable in Theano.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :return: the created tensor variable
        """
        return ext.new_tensor(
            name=name, ndim=extra_dims + 1, dtype=theano.config.floatX)
