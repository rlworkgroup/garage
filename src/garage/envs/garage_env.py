"""Wrapper class that converts gym.Env into GarageEnv."""

import copy

import akro
import gym

from garage.envs.bullet import _get_bullet_env_list, BulletEnv
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

    GarageEnv handles all environments created by gym.make().
    It returns a different wrapper class instance if the input environment
    requires special handling.
    Current supported wrapper classes are:
        garage.envs.bullet.BulletEnv for Bullet-based gym environments.
    See __new__() for details.
    """

    def __new__(cls, *args, **kwargs):
        """Returns environment specific wrapper based on input environment type.

        Args:
            args: positional arguments
            kwargs: keyword arguments

        Returns:
             garage.envs.bullet.BulletEnv: if the environment is a bullet-based
                environment. Else returns a garage.envs.GarageEnv
        """
        # Determine if the input env is a bullet-based gym environment
        env = None
        if 'env' in kwargs:  # env passed as a keyword arg
            env = kwargs['env']
        elif len(args) > 1 and args[0]:  # env passed as a positional arg
            env = args[0]
        if env and any(env.env.spec.id == name
                       for name in _get_bullet_env_list()):
            return BulletEnv(env)

        env_name = ''
        if 'env_name' in kwargs:  # env_name as a keyword arg
            env_name = kwargs['env_name']
        elif len(args) > 2 and args[1] != '':  # env_name as a positional arg
            env_name = args[1]
        if env_name != '' and any(env_name == name
                                  for name in _get_bullet_env_list()):
            return BulletEnv(gym.make(env_name))

        return super(GarageEnv, cls).__new__(cls)

    def __init__(self, env=None, env_name='', is_image=False):
        """Initializes a GarageEnv.

        Args:
            env (gym.wrappers.time_limit): A gym.wrappers.time_limit.TimeLimit
                object wrapping a gym.Env created via gym.make().
            env_name (str): If the env_name is speficied, a gym environment
                with that name will be created. If such an environment does not
                exist, a `gym.error` is thrown.
            is_image (bool): True if observations contain pixel values,
                false otherwise. Setting this to true converts a gym.Spaces.Box
                obs space to an akro.Image and normalizes pixel values.
        """
        # Needed for deserialization
        self._env_name = env_name
        self._env = env

        if env_name:
            super().__init__(gym.make(env_name))
        else:
            super().__init__(env)

        self.action_space = akro.from_gym(self.env.action_space)
        self.observation_space = akro.from_gym(self.env.observation_space,
                                               is_image=is_image)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space)

    @property
    def spec(self):
        """Return the environment specification.

        This property needs to exist, since it's defined as a property in
        gym.Wrapper in a way that makes it difficult to overwrite.

        Returns:
            garage.envs.env_spec.EnvSpec: The envionrment specification.

        """
        return self._spec

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
        # pylint: disable=import-outside-toplevel
        if self.env.spec:
            if any(package in getattr(self.env.spec, 'entry_point', '')
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
            elif any(package in getattr(self.env.spec, 'entry_point', '')
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
            action (np.ndarray): An action provided by the agent.

        Returns:
            np.ndarray: Agent's observation of the current environment
            float: Amount of reward returned after previous action
            bool: Whether the episode has ended, in which case further step()
                calls will return undefined results
            dict: Contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning)

        """
        observation, reward, done, info = self.env.step(action)
        # gym envs that are wrapped in TimeLimit wrapper modify
        # the done/termination signal to be true whenever a time
        # limit expiration occurs. The following statement sets
        # the done signal to be True only if caused by an
        # environment termination, and not a time limit
        # termination. The time limit termination signal
        # will be saved inside env_infos as
        # 'GarageEnv.TimeLimitTerminated'
        if 'TimeLimit.truncated' in info:
            info['GarageEnv.TimeLimitTerminated'] = done  # done = True always
            done = not info['TimeLimit.truncated']
        return observation, reward, done, info

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
