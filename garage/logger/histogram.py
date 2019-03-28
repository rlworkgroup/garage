"""Histogram logger input."""
import numpy as np


class Histogram(np.ndarray):
    """A `garage.logger` input representing a histogram of raw data.

    This is implemented as a typed view of a numpy array. It will accept
    input that `numpy.asarray` will.

    See https://docs.scipy.org/doc/numpy/user/basics.subclassing.html for
    details on implementation.
    """

    def __new__(cls, *args, **kwargs):
        """Reimplement `numpy.ndarray.__new__`.

        Creates objects of this class using `numpy.asarray`, then view-casts
        them back into the class `Histogram`.
        """
        return np.asarray(*args, **kwargs).view(cls)
