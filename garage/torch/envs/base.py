"""Wrapper class that converts gym.Env into TorchEnv."""
from cached_property import cached_property
from gym.spaces import Box as GymBox
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple

from garage.envs import GarageEnv
from garage.envs import EnvSpec
from garage.misc.overrides import overrides
from garage.spaces import Box
from garage.spaces import Dict
from garage.spaces import Discrete
from garage.spaces import Product


class TorchEnv(GarageEnv):
    """
    Returns a Torch wrapper class for gym.Env.

    Args:
        env (gym.Env): the env that will be wrapped
    """

    def __init__(self, env):
        super(TorchEnv, self).__init__(env)
        self.action_space = self._to_garage_space(self.env.action_space)
        self.observation_space = self._to_garage_space(
            self.env.observation_space)

    def _to_garage_space(self, space):
        """
        Converts a gym.space to a garage.torch.space.

        Returns:
            space (garage.torch.spaces)
        """
        if isinstance(space, GymBox):
            return Box(low=space.low, high=space.high)
        elif isinstance(space, GymDict):
            return Dict(space.spaces)
        elif isinstance(space, GymDiscrete):
            return Discrete(space.n)
        elif isinstance(space, GymTuple):
            return Product(list(map(self._to_garage_space, space.spaces)))
        else:
            raise NotImplementedError

    @cached_property
    @overrides
    def spec(self):
        """
        Returns an EnvSpec.

        Returns:
            spec (garage.envs.EnvSpec)
        """
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space)
