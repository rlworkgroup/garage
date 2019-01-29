"""Wrapper class that converts gym.Env into GarageEnv."""
import collections
import warnings

import glfw
import gym
from gym.spaces import Box as GymBox
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple

from garage.core import Parameterized
from garage.core import Serializable
from garage.envs.env_spec import EnvSpec
from garage.spaces import Box
from garage.spaces import Dict
from garage.spaces import Discrete
from garage.spaces import Tuple

# The gym environments using one of the packages in the following list as entry
# points don't close their viewer windows.
KNOWN_GYM_NOT_CLOSE_VIEWER = [
    # Please keep alphabetized
    "gym.envs.mujoco",
    "gym.envs.robotics"
]


class GarageEnv(gym.Wrapper, Parameterized, Serializable):
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

    def __init__(self, env=None, env_name=""):
        if env_name:
            super().__init__(gym.make(env_name))
        else:
            super().__init__(env)

        self.action_space = self._to_garage_space(self.env.action_space)
        self.observation_space = self._to_garage_space(
            self.env.observation_space)

        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())

    def close(self):
        """
        Close the wrapped env.

        Returns:
            None
        """
        self._close_mjviewer_window()
        self.env.close()

    def _close_mjviewer_window(self):
        """
        Close the MjViewer window.

        Unfortunately, the gym environments using MuJoCo don't close the viewer
        windows properly, which leads to "out of memory" issues when several
        of these environments are tested one after the other.
        This method searches for the viewer object of type MjViewer, and if the
        environment is wrapped in other environment classes, it performs depth
        search in those as well.
        This method can be removed once OpenAI solves the issue.
        """
        if self.env.spec:
            if any(package in self.env.spec._entry_point
                   for package in KNOWN_GYM_NOT_CLOSE_VIEWER):
                # This import is not in the header to avoid a MuJoCo dependency
                # with non-MuJoCo environments that use this base class.
                from mujoco_py.mjviewer import MjViewer
                if (hasattr(self.env, "viewer")
                        and isinstance(self.env.viewer, MjViewer)):
                    glfw.destroy_window(self.env.viewer.window)
                else:
                    env_itr = self.env
                    while hasattr(env_itr, "env"):
                        env_itr = env_itr.env
                        if (hasattr(env_itr, "viewer")
                                and isinstance(env_itr.viewer, MjViewer)):
                            glfw.destroy_window(env_itr.viewer.window)
                            break

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
        Returns an EnvSpec with garage.spaces.

        Returns:
            spec (garage.envs.EnvSpec)
        """
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space)

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

    def _to_garage_space(self, space):
        """
        Converts a gym.space into a garage.space.

        Args:
            space (gym.spaces)

        Returns:
            space (garage.spaces)
        """
        if isinstance(space, GymBox):
            return Box(low=space.low, high=space.high, dtype=space.dtype)
        elif isinstance(space, GymDict):
            return Dict(space.spaces)
        elif isinstance(space, GymDiscrete):
            return Discrete(space.n)
        elif isinstance(space, GymTuple):
            return Tuple(list(map(self._to_garage_space, space.spaces)))
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
