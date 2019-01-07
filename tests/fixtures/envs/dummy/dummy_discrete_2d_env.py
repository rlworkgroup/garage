import gym
import numpy as np


class DummyDiscrete2DEnv(gym.Env):
    """A dummy discrete environment."""

    @property
    def observation_space(self):
        """Return an observation space."""
        self.shape = (2, 2)
        return gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=self.shape, dtype=np.float32)

    @property
    def action_space(self):
        """Return an action space."""
        return gym.spaces.Discrete(2)

    def reset(self):
        """Reset the environment."""
        return np.zeros(self.shape)

    def step(self, action):
        """Step the environment."""
        return self.observation_space.sample(), 0, True, dict()
