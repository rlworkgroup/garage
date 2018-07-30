"""Wrapper class that converts gym.Env into TheanoEnv."""
from cached_property import cached_property

from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple

from garage.envs import GarageEnv
from garage.misc.overrides import overrides
from garage.theano.spaces import Box
from garage.theano.spaces import Discrete
from garage.theano.spaces import Product


class TheanoEnv(GarageEnv):
    """
    Returns a Theano wrapper class for gym.Env.

    Args:
        env (gym.Env): the env that will be wrapped
    """

    def __init__(self, env):
        super(TheanoEnv, self).__init__(env)

    def _to_garage_space(self, space):
        """
        Converts gym.space to a Theano space.

        Returns:
            space (garage.theano.spaces)
        """
        if isinstance(space, GymBox):
            return Box(low=space.low, high=space.high)
        elif isinstance(space, GymDiscrete):
            return Discrete(space.n)
        elif isinstance(space, GymTuple):
            return Product(list(map(self._to_theano_space, space.spaces)))
        else:
            raise NotImplementedError

    @cached_property
    @overrides
    def action_space(self):
        """Returns a converted action_space."""
        return self._to_garage_space(self.action_space)

    @cached_property
    @overrides
    def observation_space(self):
        """Returns a converted observation_space."""
        return self._to_garage_space(self.observation_space)
