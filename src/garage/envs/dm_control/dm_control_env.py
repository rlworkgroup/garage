"""DM control environment."""
import math

import akro
from dm_control import suite
from dm_control.rl.control import flatten_observation
from dm_env import StepType as dm_StepType
import numpy as np

from garage import Environment, StepType, TimeStep
from garage.envs import EnvSpec
from garage.envs.dm_control.dm_control_viewer import DmControlViewer


def _flat_shape(observation):
    """Returns the flattend shape of observation.

    Args:
        observation (np.ndarray): the observation

    Returns:
        np.ndarray: the flattened dimension of observation.
    """
    return np.sum(int(np.prod(v.shape)) for k, v in observation.items())


class DmControlEnv(Environment):
    """Binding for `dm_control <https://arxiv.org/pdf/1801.00690.pdf>`."""

    def __init__(self, env, name=None, max_episode_length=math.inf):
        """Create a DmControlEnv.

        Args:
            env (dm_control.suite.Task): The wrapped dm_control environment.
            name (str): Name of the environment.
            max_episode_length (int): The maximum steps allowed for an episode.
        """
        self._name = name or type(env.task).__name__
        self._env = env
        self._viewer = None
        self._last_observation = None
        self._step_cnt = 0
        self._max_episode_length = max_episode_length

        # action space
        action_spec = self._env.action_spec()
        if (len(action_spec.shape) == 1) and (-np.inf in action_spec.minimum or
                                              np.inf in action_spec.maximum):
            self._action_space = akro.Discrete(np.prod(action_spec.shape))
        else:
            self._action_space = akro.Box(low=action_spec.minimum,
                                          high=action_spec.maximum,
                                          dtype=np.float32)

        # observation_space
        flat_dim = _flat_shape(self._env.observation_spec())
        self._observation_space = akro.Box(low=-np.inf,
                                           high=np.inf,
                                           shape=[flat_dim],
                                           dtype=np.float32)

        # spec
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
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return ['rgb_array']

    @classmethod
    def from_suite(cls, domain_name, task_name, max_episode_length=math.inf):
        """Create a DmControl task given the domain name and task name.

        Args:
            domain_name (str): Domain name
            task_name (str): Task name

        Return:
            dm_control.suit.Task: the dm_control task environment
        """
        return cls(env=suite.load(domain_name, task_name),
                   name='{}.{}'.format(domain_name, task_name),
                   max_episode_length=max_episode_length)

    def reset(self):
        """Resets the environment.

        Returns:
            numpy.ndarray: The first observation. It must conforms to
            `observation_space`.
            dict: The episode-level information. Note that this is not part
            of `env_info` provided in `step()`. It contains information of
            the entire episode， which could be needed to determine the first
            action (e.g. in the case of goal-conditioned or MTRL.)

        """
        time_step = self._env.reset()
        first_obs = flatten_observation(time_step.observation)['observations']

        self._step_cnt = 0
        self._last_observation = first_obs
        # Populate episode_info if needed.
        episode_info = {}
        return first_obs, episode_info

    def step(self, action):
        """Steps the environment with the action and returns a `TimeStep`.

        Args:
            action (object): input action

        Returns:
            TimeStep: The time step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.
        """
        if self._last_observation is None:
            raise RuntimeError('reset() must be called before step()!')

        dm_time_step = self._env.step(action)
        if self._viewer:
            self._viewer.render()

        observation = flatten_observation(
            dm_time_step.observation)['observations']

        last_obs = self._last_observation
        self._step_cnt += 1
        self._last_observation = observation

        # Determine step type
        step_type = None
        if dm_time_step.step_type == dm_StepType.MID:
            step_type = StepType.TIMEOUT if \
                self._step_cnt >= self._max_episode_length else StepType.MID
        elif dm_time_step.step_type == dm_StepType.LAST:
            step_type = StepType.TERMINAL

        return TimeStep(
            env_spec=self.spec,
            observation=last_obs,
            action=action,
            reward=dm_time_step.reward,
            next_observation=observation,
            env_info=dm_time_step.observation,  # TODO: what to put in env_info
            agent_info={},  # TODO: can't be populated by env
            step_type=step_type)

    def render(self, mode):
        """Render the environment.

        Args:
            mode (str): render mode.

        Returns:
            np.ndarray: if mode is 'rgb_array', else return None.

        Raises:
            ValueError: if mode is not supported.
        """
        # pylint: disable=inconsistent-return-statements
        if mode == 'rgb_array':
            return self._env.physics.render()
        else:
            raise ValueError('Supported render modes are {}, but '
                             'got render mode {} instead.'.format(
                                 self.render_modes, mode))

    def visualize(self):
        """Creates a visualization of the environment."""
        if not self._viewer:
            title = 'dm_control {}'.format(self._name)
            self._viewer = DmControlViewer(title=title)
            self._viewer.launch(self._env)

    def close(self):
        """Close the environment."""
        if self._viewer:
            self._viewer.close()
        self._env.close()
        self._viewer = None
        self._env = None

    def __getstate__(self):
        """See `Object.__getstate__`.

        Returns:
            dict: dict of the class.
        """
        d = self.__dict__.copy()
        d['_viewer'] = None
        return d
