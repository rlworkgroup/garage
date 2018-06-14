import numpy as np
import tensorflow as tf

from garage.misc import ext
from garage.misc import special
from garage.spaces import Space


class Discrete(Space):
    """
    {0,1,...,n-1}
    """

    def __init__(self, n):
        self._n = n

    @property
    def n(self):
        return self._n

    def sample(self):
        return np.random.randint(self.n)

    def sample_n(self, n):
        return np.random.randint(low=0, high=self.n, size=n)

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == () and x.dtype.kind == 'i' and x >= 0 and x < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        return self.n == other.n

    def flatten(self, x):
        return special.to_onehot(x, self.n)

    def unflatten(self, x):
        return special.from_onehot(x)

    def flatten_n(self, x):
        return special.to_onehot_n(x, self.n)

    def unflatten_n(self, x):
        return special.from_onehot_n(x)

    @property
    def default_value(self):
        return 0

    @property
    def flat_dim(self):
        return self.n

    def weighted_sample(self, weights):
        return special.weighted_sample(weights, range(self.n))

    def new_tensor_variable(self, name, extra_dims):
        # needed for safe conversion to float32
        return tf.placeholder(
            dtype=tf.uint8,
            shape=[None] * extra_dims + [self.flat_dim],
            name=name)

    @property
    def dtype(self):
        return tf.uint8

    def __eq__(self, other):
        if not isinstance(other, Discrete):
            return False
        return self.n == other.n

    def __hash__(self):
        return hash(self.n)
