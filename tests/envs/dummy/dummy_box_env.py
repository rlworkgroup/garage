import gym
import numpy as np


class DummyBoxEnv(gym.Env):
    """A dummy box environment."""

    @property
    def observation_space(self):
        """Return an observation space."""
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, ), dtype=np.float32)

    @property
    def action_space(self):
        """Return an action space."""
        return gym.spaces.Box(
            low=-5.0, high=5.0, shape=(1, ), dtype=np.float32)

    def reset(self):
        """Reset the environment."""
        return np.zeros(1)

    def step(self, action):
        """Step the environment."""
        return np.zeros(1), 0, True, dict()
