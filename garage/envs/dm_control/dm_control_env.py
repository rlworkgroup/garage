from dm_control import suite
from dm_control.rl.control import flatten_observation
from dm_control.rl.environment import StepType
import gym
import numpy as np

from garage.core import Serializable
from garage.envs import Step
from garage.envs.dm_control import DmControlViewer


class DmControlEnv(gym.Env, Serializable):
    """
    Binding for [dm_control](https://arxiv.org/pdf/1801.00690.pdf)
    """

    def __init__(
            self,
            domain_name,
            task_name,
            plot=False,
            width=320,
            height=240,
    ):
        self._env = suite.load(domain_name=domain_name, task_name=task_name)

        self._total_reward = 0
        self._render_kwargs = {'width': width, 'height': height}

        if plot:
            self._viewer = DmControlViewer()
        else:
            self._viewer = None

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    def step(self, action):
        time_step = self._env.step(action)
        if time_step.reward:
            self._total_reward += time_step.reward
        return Step(
            flatten_observation(time_step.observation)["observations"],
            time_step.reward, time_step.step_type == StepType.LAST,
            **time_step.observation)

    def reset(self):
        self._total_reward = 0
        time_step = self._env.reset()
        return flatten_observation(time_step.observation)["observations"]

    def render(self):
        if self._viewer:
            pixels_img = self._env.physics.render(**self._render_kwargs)
            self._viewer.loop_once(pixels_img)

    def close(self):
        if self._viewer:
            self._viewer.finish()

    def _flat_shape(self, observation):
        return np.sum(int(np.prod(v.shape)) for k, v in observation.items())

    @property
    def action_space(self):
        action_spec = self._env.action_spec()
        if (len(action_spec.shape) == 1) and (-np.inf in action_spec.minimum or
                                              np.inf in action_spec.maximum):
            return gym.spaces.Discrete(np.prod(action_spec.shape))
        else:
            return gym.spaces.Box(
                action_spec.minimum, action_spec.maximum, dtype=np.float32)

    @property
    def observation_space(self):
        flat_dim = self._flat_shape(self._env.observation_spec())
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=[flat_dim], dtype=np.float32)

    @property
    def total_reward(self):
        return self._total_reward
