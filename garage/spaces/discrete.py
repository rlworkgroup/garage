import numpy as np

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

    def contains(self, x):
        x = np.asarray(x)
        return x.shape == () and x.dtype.kind == 'i' and x >= 0 and x < self.n

    def __repr__(self):
        return "Discrete(%d)" % self.n

    def __eq__(self, other):
        if not isinstance(other, Discrete):
            return False
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
    def flat_dim(self):
        return self.n

    def weighted_sample(self, weights):
        return special.weighted_sample(weights, range(self.n))

    @property
    def default_value(self):
        return 0

    def __hash__(self):
        return hash(self.n)

    def new_tensor_variable(self, name, extra_dims):
        raise NotImplementedError
