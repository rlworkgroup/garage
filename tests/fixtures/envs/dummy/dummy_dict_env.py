import gym
import numpy as np

from garage.envs import EnvSpec
from tests.fixtures.envs.dummy import DummyEnv


class DummyDictEnv(DummyEnv):
    """A dummy dict environment."""

    def __init__(self, random=True):
        super().__init__(random)
        self.spec = EnvSpec(
            action_space=self.action_space,
            observation_space=self.observation_space)

    @property
    def observation_space(self):
        """Return an observation space."""

        return gym.spaces.Dict({
            'achieved_goal':
            gym.spaces.Box(
                low=-200., high=200., shape=(3, ), dtype=np.float32),
            'desired_goal':
            gym.spaces.Box(
                low=-200., high=200., shape=(3, ), dtype=np.float32),
            'observation':
            gym.spaces.Box(
                low=-200., high=200., shape=(25, ), dtype=np.float32)
        })

    @property
    def action_space(self):
        """Return an action space."""
        return gym.spaces.Dict({
            'action':
            gym.spaces.Box(low=-5.0, high=5.0, shape=(1, ), dtype=np.float32)
        })

    def reset(self):
        """Reset the environment."""
        return self.observation_space.sample()

    def step(self, action):
        """Step the environment."""
        return self.observation_space.sample(), 0, True, dict()
