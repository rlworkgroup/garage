"""Wrapper class that converts gym.Env into GarageEnv."""
from cached_property import cached_property
import collections
import warnings

import gym
from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple

from garage.core import Parameterized
from garage.core import Serializable
from garage.envs.env_spec import EnvSpec
from garage.misc.overrides import overrides
from garage.spaces import Box
from garage.spaces import Discrete
from garage.spaces import Product


class GarageEnv(gym.Wrapper, Parameterized, Serializable):
    """
    Returns a Garage wrapper class for gym.Env.

    In order to provide pickling (serialization) and parameterization
    for gym.Envs, they must be wrapped with GarageEnv. This ensures
    compatibility with existing samplers and checkpointing when the
    envs are passed internally around garage.

    Furthermore, GarageEnv silently converts action_space and
    observation_space from gym.Spaces to garage.spaces.

    Args:
        env (gym.Env): the env that will be wrapped
    """

    def __init__(self, env):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)
        super().__init__(env)

    @cached_property
    @overrides
    def action_space(self):
        """
        Classes inheriting from GarageEnv need to convert
        action_space from gym.space to garage.space.

        Returns:
            NotImplementedError
        """
        return self._to_garage_space(self.env.action_space)

    @cached_property
    @overrides
    def observation_space(self):
        """
        Classes inheriting from GarageEnv need to convert
        observation_space from gym.space to garage.space.

        Returns:
            NotImplementedError
        """
        return self._to_garage_space(self.env.observation_space)

    def close(self):
        """
        Close the wrapped env.

        Returns:
            None
        """
        self.env.close()

    def get_params_internal(self, **tags):
        """
        Returns an empty list if env.get_params() is called.

        Returns:
            An empty list
        """
        warnings.warn("get_params_internal is deprecated", DeprecationWarning)
        return []

    @property
    def horizon(self):
        """
        Get the maximum episode steps for the wrapped env.

        Returns:
            max_episode_steps (int)
        """
        if self.env.spec is not None:
            return self.env.spec.max_episode_steps
        else:
            return NotImplementedError

    def log_diagnostics(self, paths, *args, **kwargs):
        """No env supports this function call."""
        warnings.warn("log_diagnostics is deprecated", DeprecationWarning)
        pass

    @property
    def spec(self):
        """
        Returns an EnvSpec.

        Returns:
            spec (garage.envs.EnvSpec)
        """
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    def reset(self, **kwargs):
        """
        Inheriting gym.Wrapper requires implementing this method.

        Calls reset on wrapped env.
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Inheriting gym.Wrapper requires implementing this method.

        Calls step on wrapped env.
        """
        return self.env.step(action)

    def _to_garage_space(self, space):
        """
        Converts a gym space into a garage space.

        Args:
            space (gym.spaces)

        Returns:
            space (garage.spaces)
        """
        if isinstance(space, GymBox):
            return Box(low=space.low, high=space.high)
        elif isinstance(space, GymDiscrete):
            return Discrete(space.n)
        elif isinstance(space, GymTuple):
            return Product(list(map(self._to_tf_space, space.spaces)))
        else:
            raise NotImplementedError


def Step(observation, reward, done, **kwargs):  # noqa: N802
    """
    Convenience method for creating a namedtuple from the results of
    environment.step(action). Provides the option to put extra
    diagnostic info in the kwargs (if it exists) without demanding
    an explicit positional argument.
    """
    return _Step(observation, reward, done, kwargs)


_Step = collections.namedtuple("Step",
                               ["observation", "reward", "done", "info"])
