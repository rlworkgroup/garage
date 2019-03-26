import gym
import numpy as np

from tests.fixtures.envs.dummy import DummyEnv


class DummyDiscrete2DEnv(DummyEnv):
    """A dummy discrete environment."""

    def __init__(self, random=True):
        super().__init__(random)
        self.shape = (2, 2)
        self._observation_space = gym.spaces.Box(
            low=-1, high=1, shape=self.shape, dtype=np.float32)

    @property
    def observation_space(self):
        """Return an observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    @property
    def action_space(self):
        """Return an action space."""
        return gym.spaces.Discrete(2)

    def reset(self):
        """Reset the environment."""
        self.state = np.ones(self.shape)
        return self.state

    def step(self, action):
        """Step the environment."""
        if self.state is not None:
            if self.random:
                obs = self.observation_space.sample()
            else:
                obs = self.state + action / 10.
        else:
            raise RuntimeError(
                "DummyEnv: reset() must be called before step()!")
        return obs, 0, True, dict()
