"""DM control environment."""

from dm_control import suite
from dm_control.rl.control import flatten_observation
from dm_env import StepType
import gym
import numpy as np

from garage.envs import Step
from garage.envs.dm_control.dm_control_viewer import DmControlViewer


def _flat_shape(observation):
    """Returns the flattend shape of observation.

    Args:
        observation (np.ndarray): the observation

    Returns:
        np.ndarray: the flattened dimension of observation.
    """
    return np.sum(int(np.prod(v.shape)) for k, v in observation.items())


class DmControlEnv(gym.Env):
    """Binding for `dm_control <https://arxiv.org/pdf/1801.00690.pdf>`."""

    def __init__(self, env, name=None):
        """Create a DmControlEnv.

        Args:
            env (dm_control.suite.Task): The wrapped dm_control environment.
            name (str): Name of the environment.
        """
        self._name = name or type(env.task).__name__
        self._env = env
        self._viewer = None

    @classmethod
    def from_suite(cls, domain_name, task_name):
        """Create a DmControl task given the domain name and task name.

        Args:
            domain_name (str): Domain name
            task_name (str): Task name

        Return:
            dm_control.suit.Task: the dm_control task environment
        """
        return cls(suite.load(domain_name, task_name),
                   name='{}.{}'.format(domain_name, task_name))

    def step(self, action):
        """Step the environment.

        Args:
            action (object): input action

        Returns:
            Step: The time step after applying this action.
        """
        time_step = self._env.step(action)
        return Step(
            flatten_observation(time_step.observation)['observations'],
            time_step.reward, time_step.step_type == StepType.LAST,
            **time_step.observation)

    def reset(self):
        """Reset the environment.

        Returns:
            Step: The first time step.
        """
        time_step = self._env.reset()
        return flatten_observation(time_step.observation)['observations']

    def render(self, mode='human'):
        """Render the environment.

        Args:
            mode (str): render mode.

        Returns:
            np.ndarray: if mode is 'rgb_array', else return None.

        Raises:
            ValueError: if mode is not supported.
        """
        # pylint: disable=inconsistent-return-statements
        if mode == 'human':
            if not self._viewer:
                title = 'dm_control {}'.format(self._name)
                self._viewer = DmControlViewer(title=title)
                self._viewer.launch(self._env)
            self._viewer.render()
            return None
        elif mode == 'rgb_array':
            return self._env.physics.render()
        else:
            raise ValueError

    def close(self):
        """Close the environment."""
        if self._viewer:
            self._viewer.close()
        self._env.close()
        self._viewer = None
        self._env = None

    @property
    def action_space(self):
        """gym.Space: the action space specification."""
        action_spec = self._env.action_spec()
        if (len(action_spec.shape) == 1) and (-np.inf in action_spec.minimum or
                                              np.inf in action_spec.maximum):
            return gym.spaces.Discrete(np.prod(action_spec.shape))
        else:
            return gym.spaces.Box(action_spec.minimum,
                                  action_spec.maximum,
                                  dtype=np.float32)

    @property
    def observation_space(self):
        """gym.Space: the observation space specification."""
        flat_dim = _flat_shape(self._env.observation_spec())
        return gym.spaces.Box(low=-np.inf,
                              high=np.inf,
                              shape=[flat_dim],
                              dtype=np.float32)

    def __getstate__(self):
        """See `Object.__getstate__`.

        Returns:
            dict: dict of the class.
        """
        d = self.__dict__.copy()
        d['_viewer'] = None
        return d
