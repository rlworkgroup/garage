"""Spaces.Discrete for Theano."""
from garage.spaces import Discrete as GarageDiscrete
from garage.theano.misc import tensor_utils


class Discrete(GarageDiscrete):
    """Theano extension of garage.Discrete."""

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable in Theano.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :return: the created tensor variable
        """
        if self.n <= 2**8:
            return tensor_utils.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint8')
        elif self.n <= 2**16:
            return tensor_utils.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint16')
        else:
            return tensor_utils.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint32')
