"""Wrapper class that converts gym.Env into GarageEnv."""
import collections

import akro
import gym

from garage.core import Serializable
from garage.envs.env_spec import EnvSpec

# The gym environments using one of the packages in the following lists as
# entry points don't close their viewer windows.
KNOWN_GYM_NOT_CLOSE_VIEWER = [
    # Please keep alphabetized
    'gym.envs.atari',
    'gym.envs.box2d',
    'gym.envs.classic_control'
]

KNOWN_GYM_NOT_CLOSE_MJ_VIEWER = [
    # Please keep alphabetized
    'gym.envs.mujoco',
    'gym.envs.robotics'
]


class GarageEnv(gym.Wrapper, Serializable):
    """
    Returns an abstract Garage wrapper class for gym.Env.

    In order to provide pickling (serialization) and parameterization
    for gym.Envs, they must be wrapped with a GarageEnv. This ensures
    compatibility with existing samplers and checkpointing when the
    envs are passed internally around garage.

    Furthermore, classes inheriting from GarageEnv should silently
    convert action_space and observation_space from gym.Spaces to
    akro.spaces.

    Args: env (gym.Env): the env that will be wrapped
    """

    def __init__(self, env=None, env_name=''):
        if env_name:
            super().__init__(gym.make(env_name))
        else:
            super().__init__(env)

        self.action_space = akro.from_gym(self.env.action_space)
        self.observation_space = akro.from_gym(self.env.observation_space)
        if self.spec:
            self.spec.action_space = self.action_space
            self.spec.observation_space = self.observation_space
        else:
            self.spec = EnvSpec(action_space=self.action_space,
                                observation_space=self.observation_space)

        Serializable.quick_init(self, locals())

    def close(self):
        """
        Close the wrapped env.

        Returns:
            None

        """
        self._close_viewer_window()
        self.env.close()

    def _close_viewer_window(self):
        """
        Close viewer window.

        Unfortunately, some gym environments don't close the viewer windows
        properly, which leads to "out of memory" issues when several of
        these environments are tested one after the other.
        This method searches for the viewer object of type MjViewer, Viewer
        or SimpleImageViewer, based on environment, and if the environment
        is wrapped in other environment classes, it performs depth search
        in those as well.
        This method can be removed once OpenAI solves the issue.
        """
        if self.env.spec:
            if any(package in self.env.spec._entry_point
                   for package in KNOWN_GYM_NOT_CLOSE_MJ_VIEWER):
                # This import is not in the header to avoid a MuJoCo dependency
                # with non-MuJoCo environments that use this base class.
                try:
                    from mujoco_py.mjviewer import MjViewer
                    import glfw
                except ImportError:
                    # If we can't import mujoco_py, we must not have an
                    # instance of a class that we know how to close here.
                    return
                if (hasattr(self.env, 'viewer')
                        and isinstance(self.env.viewer, MjViewer)):
                    glfw.destroy_window(self.env.viewer.window)
            elif any(package in self.env.spec._entry_point
                     for package in KNOWN_GYM_NOT_CLOSE_VIEWER):
                if hasattr(self.env, 'viewer'):
                    from gym.envs.classic_control.rendering import (
                        Viewer, SimpleImageViewer)
                    if (isinstance(self.env.viewer, Viewer)
                            or isinstance(self.env.viewer, SimpleImageViewer)):
                        self.env.viewer.close()

    def reset(self, **kwargs):
        """Call reset on wrapped env.

        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """Call step on wrapped env.

        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.
        """
        return self.env.step(action)


def Step(observation, reward, done, **kwargs):  # noqa: N802
    """Create a namedtuple from the results of environment.step(action).

    Provides the option to put extra diagnostic info in the kwargs (if it
    exists) without demanding an explicit positional argument.
    """
    return _Step(observation, reward, done, kwargs)


_Step = collections.namedtuple('Step',
                               ['observation', 'reward', 'done', 'info'])
