"""Wrapper class that converts gym.Env into GarageEnv."""
from abc import ABCMeta, abstractmethod
import collections
import warnings

import gym

from garage.core import Parameterized
from garage.core import Serializable


class GarageEnv(gym.Wrapper, Parameterized, Serializable, metaclass=ABCMeta):
    """
    Returns an abstract Garage wrapper class for gym.Env.

    In order to provide pickling (serialization) and parameterization
    for gym.Envs, they must be wrapped with a GarageEnv. This ensures
    compatibility with existing samplers and checkpointing when the
    envs are passed internally around garage.

    Furthermore, classes inheriting from GarageEnv should silently
    convert action_space and observation_space from gym.Spaces to
    garage.spaces.

    Args: env (gym.Env): the env that will be wrapped
    """

    def __init__(self, env):
        Serializable.quick_init(self, locals())
        Parameterized.__init__(self)
        super().__init__(env)

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

    @abstractmethod
    def spec(self):
        """
        Returns an EnvSpec with garage.spaces.

        Returns:
            spec (garage.envs.EnvSpec)
        """
        raise NotImplementedError

    def reset(self, **kwargs):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls reset on wrapped env.
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Calls step on wrapped env.
        """
        return self.env.step(action)

    @abstractmethod
    def _to_garage_space(self, space):
        """
        Converts a gym.space into a garage.space.

        Args:
            space (gym.spaces)

        Returns:
            space (garage.spaces)
        """
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
