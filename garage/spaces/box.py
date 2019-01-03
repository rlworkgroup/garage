import warnings

import numpy as np

from garage.spaces import Space


class Box(Space):
    """
    A box in R^n.
    I.e., each coordinate is bounded.
    """

    def __init__(self, low, high, shape=None, dtype=None):
        """
        Two kinds of valid input:
            Box(-1.0, 1.0, (3,4)) # low and high are scalars, and shape is
            provided
            Box(np.array([-1.0,-2.0]), np.array([2.0,4.0])) # low and high are
            arrays of the same shape

        If dtype is not specified, we assume dtype to be np.float32,
        but when low=0 and high=255, it is very likely to be np.uint8.
        We autodetect this case and warn user. It is different from gym.Box,
        where they warn user as long as dtype is not specified.
        """
        if shape is None:
            assert low.shape == high.shape
            self.low = low
            self.high = high
        else:
            assert np.isscalar(low) and np.isscalar(high)
            self.low = low + np.zeros(shape)
            self.high = high + np.zeros(shape)

        if dtype is None:
            if (self.low == 0).all() and (self.high == 255).all():
                self.dtype = np.uint8
                warnings.warn("garage.spaces.Box detected dtype as np.uint8.\
                    Please provide explicit dtype.")
            else:
                self.dtype = np.float32

        else:
            self.dtype = dtype

    def sample(self):
        return np.random.uniform(
            low=self.low, high=self.high,
            size=self.low.shape).astype(self.dtype)

    def contains(self, x):
        return x.shape == self.shape and (x >= self.low).all() and (
            x <= self.high).all()

    @property
    def shape(self):
        return self.low.shape

    @property
    def flat_dim(self):
        return np.prod(self.low.shape)

    @property
    def bounds(self):
        return self.low, self.high

    def flatten(self, x):
        return np.asarray(x).flatten()

    def unflatten(self, x):
        return np.asarray(x).reshape(self.shape)

    def flatten_n(self, xs):
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0], -1))

    def unflatten_n(self, xs):
        xs = np.asarray(xs)
        return xs.reshape((xs.shape[0], ) + self.shape)

    def __repr__(self):
        return "Box" + str(self.shape)

    def __eq__(self, other):
        return isinstance(other, Box) \
            and np.allclose(self.low, other.low) \
            and np.allclose(self.high, other.high)

    def __hash__(self):
        return hash((self.low, self.high))

    def new_tensor_variable(self, name, extra_dims):
        raise NotImplementedError
