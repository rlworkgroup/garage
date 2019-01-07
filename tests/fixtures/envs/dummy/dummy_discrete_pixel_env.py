import gym
import numpy as np


class DummyDiscretePixelEnv(gym.Env):
    """A dummy discrete environment."""

    @property
    def observation_space(self):
        """Return an observation space."""
        self.shape = (10, 10, 3)
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8)
        return self._observation_space

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
