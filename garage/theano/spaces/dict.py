"""Spaces.Dict for Theano."""
from collections import Iterable
from collections import OrderedDict

import gym
from gym.spaces import Box as GymBox
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple
import numpy as np

from garage.misc.overrides import overrides
from garage.spaces.dict import Dict as GarageDict
from garage.theano.spaces.box import Box
from garage.theano.spaces.discrete import Discrete
from garage.theano.spaces.tuple import Tuple


class Dict(GarageDict):
    """Theano extension of garage.Dict."""

    def __init__(self, spaces):
        """
        Initialize garage.Dict.

        Args:
            spaces (dict/list)
        """
        assert spaces is not None
        super().__init__(spaces)

        # Convert each space to a garage.space
        for key, space in self.spaces.items():
            self.spaces[key] = self._to_garage_space(space)

        # Retrieve dimension length for un/flatten_n
        self._dims = [space.flat_dim for space in self.spaces.values()]

    @property
    def dtype(self):
        """
        Return the element's data type.

        Returns:
            dtype (gym.spaces)

        """
        return gym.spaces

    @overrides
    def sample(self):
        """
        Retrieve a sample from every space in the dict.

        Returns:
            OrderedDict

        """
        return OrderedDict(
            [(key, space.sample()) for key, space in self.spaces.items()])

    @property
    def flat_dim(self):
        """
        Return a flat dimension of the dict space.

        Returns:
            sum (int)

        """
        return sum([space.flat_dim for _, space in self.spaces.items()])

    def flat_dim_with_keys(self, keys: Iterable):
        """
        Return a flat dimension of the spaces specified by the keys.

        Returns:
            sum (int)

        """
        return sum([self.spaces[key].flat_dim for key in keys])

    def flatten(self, x):
        """
        Return flattened obs of all spaces using values in x.

        Returns:
            list

        """
        return np.concatenate(
            [
                space.flatten(xi)
                for space, xi in zip(self.spaces.values(), x.values())
            ],
            axis=-1)

    def unflatten(self, x):
        """
        Return unflattened obs of all spaces using values in x.

        Returns:
            OrderedDict

        """
        flat_x = np.split(x, np.cumsum(self._dims)[:-1])
        return OrderedDict([(key, self.spaces[key].unflatten(xi))
                            for key, xi in zip(self.spaces.keys(), flat_x)])

    def flatten_n(self, xs):
        """
        Return flattened obs of all spaces using values in x for x in xs.

        Returns:
            list.shape = n x np.sum(self._dims)

        """
        return [self.flatten(x) for x in xs]

    def unflatten_n(self, xs):
        """
        Return unflattened obs of all spaces using values in x for x in xs.

        Returns:
            A list of OrderedDict.

        """
        return [self.unflatten(x) for x in xs]

    def flatten_with_keys(self, x, keys: Iterable):
        """
        Return flattened obs of spaces specified by the keys using x.

        Returns:
            list

        """
        return np.concatenate(
            [
                self.spaces[key].flatten(xi)
                for key, xi in zip(self.spaces.keys(), x.values())
                if key in keys
            ],
            axis=-1)

    def unflatten_with_keys(self, x, keys: Iterable):
        """
        Return unflattened obs of spaces specified by the keys using x.

        Returns:
            OrderedDict

        """
        unflat_x = self.unflatten(x)
        return OrderedDict({key: unflat_x[key] for key in sorted(keys)})

    def _to_garage_space(self, space):
        """
        Convert a gym.space to a garage.tf.space.

        Returns:
            space (garage.tf.spaces)

        """
        if isinstance(space, GymBox):
            return Box(low=space.low, high=space.high)
        elif isinstance(space, GymDict):
            return Dict(space.spaces)
        elif isinstance(space, GymDiscrete):
            return Discrete(space.n)
        elif isinstance(space, GymTuple):
            return Tuple(list(map(self._to_garage_space, space.spaces)))
        else:
            raise NotImplementedError
