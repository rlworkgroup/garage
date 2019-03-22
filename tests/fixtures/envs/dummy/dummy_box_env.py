import gym
import numpy as np

from tests.fixtures.envs.dummy import DummyEnv


class DummyBoxEnv(DummyEnv):
    """A dummy box environment."""

    def __init__(self, random=True, obs_dim=(4, ), action_dim=(2, )):
        super().__init__(random, obs_dim, action_dim)

    @property
    def observation_space(self):
        """Return an observation space."""
        return gym.spaces.Box(
            low=-1, high=1, shape=self._obs_dim, dtype=np.float32)

    @property
    def action_space(self):
        """Return an action space."""
        return gym.spaces.Box(
            low=-5.0, high=5.0, shape=self._action_dim, dtype=np.float32)

    def reset(self):
        """Reset the environment."""
        return np.ones(self._obs_dim, dtype=np.float32)

    def step(self, action):
        """Step the environment."""
        return self.observation_space.sample(), 0, True, dict()
