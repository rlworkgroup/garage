import gym
import numpy as np

from garage.core import Serializable
from garage.envs import Step


class PointEnv(gym.Env, Serializable):
    def __init__(self, goal=None):
        if goal is None:
            self.goal = np.array([0, 0])
        else:
            self.goal = goal

        # Always call Serializable constructor last
        Serializable.quick_init(self, locals())

    @property
    def observation_space(self):
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(2, ), dtype=np.float32)

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=-0.1, high=0.1, shape=(2, ), dtype=np.float32)

    def reset(self):
        self._state = np.random.uniform(-1, 1, size=(2, ))
        observation = np.copy(self._state)
        return observation

    def step(self, action):
        self._state = self._state + action
        x, y = self._state
        reward = -((x - self.goal[0])**2 + (y - self.goal[1])**2)**0.5
        done = abs(x) < 0.01 and abs(y) < 0.01
        next_observation = np.copy(self._state)
        return Step(observation=next_observation, reward=reward, done=done)

    def render(self, mode=None):
        print('current state:', self._state)

    def log_diagnostics(self, paths):
        pass

    def reset_task(self, task):
        self.goal = task

    @property
    def task_space(self):
        return gym.spaces.Box(
            low=-10., high=10., shape=(2, ), dtype=np.float32)
