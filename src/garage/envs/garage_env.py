"""Wrapper class that converts gym.Env into GarageEnv."""

import copy
import math

import akro
import gym
from gym.wrappers.time_limit import TimeLimit
import numpy as np

from garage import Environment, StepType, TimeStep
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


class GarageEnv(Environment):
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
        # pylint: disable=import-outside-toplevel
        # Determine if the input env is a bullet-based gym environment
        env = None
        if 'env' in kwargs:  # env passed as a keyword arg
            env = kwargs['env']
        elif len(args) >= 1 and isinstance(args[0], TimeLimit):
            # env passed as a positional arg
            env = args[0]

        # get the inner env if it is a gym.Wrapper
        if env and issubclass(env.__class__, gym.Wrapper):
            env = env.unwrapped

        if env and env.spec.id.find('Bullet') >= 0:
            from garage.envs.bullet import BulletEnv
            return BulletEnv(env)

        env_name = ''
        if 'env_name' in kwargs:  # env_name as a keyword arg
            env_name = kwargs['env_name']
        elif len(args) >= 2:
            # env_name as a positional arg
            env_name = args[1]
        if env_name != '' and env_name.find('Bullet') >= 0:
            from garage.envs.bullet import BulletEnv
            return BulletEnv(gym.make(env_name))

        return super(GarageEnv, cls).__new__(cls)

    def __init__(self,
                 env=None,
                 env_name='',
                 is_image=False,
                 max_episode_length=math.inf):
        """Initializes a GarageEnv.

        Note that if `env` and `env_name` are passed in at the same time,
        `env` will be wrapped.

        Args:
            env (gym.wrappers.time_limit): A gym.wrappers.time_limit.TimeLimit
                object wrapping a gym.Env created via gym.make().
            env_name (str): If the env_name is speficied, a gym environment
                with that name will be created. If such an environment does not
                exist, a `gym.error` is thrown.
            is_image (bool): True if observations contain pixel values,
                false otherwise. Setting this to true converts a gym.Spaces.Box
                obs space to an akro.Image and normalizes pixel values.
            max_episode_length (int): The maximum steps allowed for an episode.
        """
        self.env = env if env else gym.make(env_name)

        env = self.env
        if isinstance(self.env, TimeLimit):  # env is wrapped by TimeLimit
            self.env._max_episode_steps = max_episode_length
            self._render_modes = self.env.unwrapped.metadata['render.modes']
        elif 'metadata' in env.__dict__:
            self._render_modes = env.metadata['render.modes']
        else:
            self._render_modes = []

        self._last_observation = None
        self._step_cnt = 0
        self._visualize = False

        self._action_space = akro.from_gym(self.env.action_space)
        self._observation_space = akro.from_gym(self.env.observation_space,
                                                is_image=is_image)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """garage.envs.env_spec.EnvSpec: The envionrment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return self._render_modes

    def reset(self, **kwargs):
        """Call reset on wrapped env.

        Args:
            kwargs: Keyword args

        Returns:
            numpy.ndarray: The first observation. It must conforms to
            `observation_space`.
            dict: The episode-level information. Note that this is not part
            of `env_info` provided in `step()`. It contains information of
            the entire episode， which could be needed to determine the first
            action (e.g. in the case of goal-conditioned or MTRL.)

        """
        first_obs = self.env.reset(**kwargs)

        self._step_cnt = 0
        self._last_observation = first_obs
        # Populate episode_info if needed.
        episode_info = {}
        return first_obs, episode_info

    def step(self, action):
        """Call step on wrapped env.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            TimeStep: The time step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.

        """
        if self._last_observation is None:
            raise RuntimeError('reset() must be called before step()!')

        observation, reward, done, info = self.env.step(action)

        if self._visualize:
            self.env.render(mode='human')

        last_obs = self._last_observation
        # Type conversion
        if not isinstance(reward, float):
            reward = float(reward)

        self._last_observation = observation
        self._step_cnt += 1

        step_type = None
        if done:
            step_type = StepType.TERMINAL
        elif self._step_cnt == 1:
            step_type = StepType.FIRST
        else:
            step_type = StepType.MID

        # gym envs that are wrapped in TimeLimit wrapper modify
        # the done/termination signal to be true whenever a time
        # limit expiration occurs. The following statement sets
        # the done signal to be True only if caused by an
        # environment termination, and not a time limit
        # termination. The time limit termination signal
        # will be saved inside env_infos as
        # 'GarageEnv.TimeLimitTerminated'
        if 'TimeLimit.truncated' in info or \
            self._step_cnt >= self._spec.max_episode_length:
            info['GarageEnv.TimeLimitTerminated'] = True
            step_type = StepType.TIMEOUT
        else:
            info['TimeLimit.truncated'] = False
            info['GarageEnv.TimeLimitTerminated'] = False

        return TimeStep(
            env_spec=self.spec,
            observation=last_obs,
            action=action,
            reward=reward,
            next_observation=observation,
            env_info=info,
            agent_info={},  # TODO: can't be populated by env
            step_type=step_type)

    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.
        """
        if mode not in self.render_modes:
            raise ValueError('Supported render modes are {}, but '
                             'got render mode {} instead.'.format(
                                 self.render_modes, mode))
        return self.env.render(mode)

    def visualize(self):
        """Creates a visualization of the environment."""
        self.env.render(mode='human')
        self._visualize = True

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
        self.__init__(state['env'])
