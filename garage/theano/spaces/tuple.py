"""Spaces.Tuple for Theano."""
from garage.spaces import Tuple as GarageTuple
from garage.theano.misc import tensor_utils


class Tuple(GarageTuple):
    """Theano extension of garage.Tuple."""

    def new_tensor_variable(self, name, extra_dims):
        """
        Create a tensor variable in Theano.

        :param name: name of the variable
        :param extra_dims: extra dimensions to be prepended
        :return: the created tensor variable
        """
        return tensor_utils.new_tensor(
            name=name,
            ndim=extra_dims + 1,
            dtype=self._common_dtype,
        )
