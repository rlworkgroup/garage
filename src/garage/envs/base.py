"""Wrapper class that converts gym.Env into GarageEnv."""

import collections
import copy

import akro
import gym

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


class GarageEnv(gym.Wrapper):
    """Returns an abstract Garage wrapper class for gym.Env.

    In order to provide pickling (serialization) and parameterization
    for gym.Envs, they must be wrapped with a GarageEnv. This ensures
    compatibility with existing samplers and checkpointing when the
    envs are passed internally around garage.

    Furthermore, classes inheriting from GarageEnv should silently
    convert action_space and observation_space from gym.Spaces to
    akro.spaces.

    Args:
        env (gym.Env): An env that will be wrapped
        env_name (str): If the env_name is speficied, a gym environment
            with that name will be created. If such an environment does not
            exist, a `gym.error` is thrown.

    """

    def __init__(self, env=None, env_name=''):
        # Needed for deserialization
        self._env_name = env_name
        self._env = env

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

    def close(self):
        """Close the wrapped env."""
        self._close_viewer_window()
        self.env.close()

    def _close_viewer_window(self):
        """Close viewer window.

        Unfortunately, some gym environments don't close the viewer windows
        properly, which leads to "out of memory" issues when several of
        these environments are tested one after the other.
        This method searches for the viewer object of type MjViewer, Viewer
        or SimpleImageViewer, based on environment, and if the environment
        is wrapped in other environment classes, it performs depth search
        in those as well.
        This method can be removed once OpenAI solves the issue.
        """
        # We need to do some strange things here to fix-up flaws in gym
        # pylint: disable=protected-access, import-outside-toplevel
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
                    if (isinstance(self.env.viewer,
                                   (SimpleImageViewer, Viewer))):
                        self.env.viewer.close()

    def reset(self, **kwargs):
        """Call reset on wrapped env.

        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Args:
            kwargs: Keyword args

        Returns:
            object: The initial observation.

        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """Call step on wrapped env.

        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Args:
            action (object): An action provided by the agent.

        Returns:
            object: Agent's observation of the current environment
            float : Amount of reward returned after previous action
            bool : Whether the episode has ended, in which case further step()
                calls will return undefined results
            dict: Contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning)

        """
        return self.env.step(action)

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s dictionary to be pickled.

        """
        # the viewer object is not pickleable
        # we first make a copy of the viewer
        env = self.env
        # get the inner env if it is a gym.Wrapper
        if issubclass(env.__class__, gym.Wrapper):
            env = env.unwrapped
        if 'viewer' in env.__dict__:
            _viewer = env.viewer
            # remove the viewer and make a copy of the state
            env.viewer = None
            state = copy.deepcopy(self.__dict__)
            # assign the viewer back to self.__dict__
            env.viewer = _viewer
            # the returned state doesn't have the viewer
            return state
        return self.__dict__

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(state['_env'], state['_env_name'])


def Step(observation, reward, done, **kwargs):  # noqa: N802
    """Create a namedtuple from the results of environment.step(action).

    Provides the option to put extra diagnostic info in the kwargs (if it
    exists) without demanding an explicit positional argument.

    Args:
        observation (object): Agent's observation of the current environment
        reward (float) : Amount of reward returned after previous action
        done (bool): Whether the episode has ended, in which case further
            step() calls will return undefined results
        kwargs: Keyword args

    Returns:
        collections.namedtuple: A named tuple of the arguments.

    """
    return _Step(observation, reward, done, kwargs)


_Step = collections.namedtuple('Step',
                               ['observation', 'reward', 'done', 'info'])
