from dm_control import suite
from dm_control.rl.control import flatten_observation
from dm_control.rl.environment import StepType
import gym
import numpy as np

from garage.envs import Step
from garage.envs.dm_control.dm_control_viewer import DmControlViewer


class DmControlEnv(gym.Env):
    """
    Binding for `dm_control <https://arxiv.org/pdf/1801.00690.pdf>`_
    """

    def __init__(self, env, name=None):
        self._name = name or type(env.task).__name__
        self._env = env
        self._viewer = None

    @classmethod
    def from_suite(cls, domain_name, task_name):
        return cls(suite.load(domain_name, task_name),
                   name='{}.{}'.format(domain_name, task_name))

    def step(self, action):
        time_step = self._env.step(action)
        return Step(
            flatten_observation(time_step.observation)['observations'],
            time_step.reward, time_step.step_type == StepType.LAST,
            **time_step.observation)

    def reset(self):
        time_step = self._env.reset()
        return flatten_observation(time_step.observation)['observations']

    def render(self, mode='human'):
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
            raise NotImplementedError

    def close(self):
        if self._viewer:
            self._viewer.close()
        self._env.close()
        self._viewer = None
        self._env = None

    def _flat_shape(self, observation):
        return np.sum(int(np.prod(v.shape)) for k, v in observation.items())

    @property
    def action_space(self):
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
        flat_dim = self._flat_shape(self._env.observation_spec())
        return gym.spaces.Box(low=-np.inf,
                              high=np.inf,
                              shape=[flat_dim],
                              dtype=np.float32)

    def __getstate__(self):
        d = self.__dict__.copy()
        d['_viewer'] = None
        return d
