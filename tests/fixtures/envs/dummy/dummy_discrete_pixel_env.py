import gym
import numpy as np

from tests.fixtures.envs.dummy import DummyEnv


class DummyDiscretePixelEnv(DummyEnv):
    """A dummy discrete environment."""

    def __init__(self, random=True):
        super().__init__(random)

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
        self.state = np.zeros(self.shape, dtype=np.uint8)
        return self.state

    def step(self, action):
        """
        Step the environment.

        Before gym fixed overflow issue for sample() in
        np.uint8 environment, we will handle the sampling here.
        We need high=256 since np.random.uniform sample from [low, high)
        (includes low, but excludes high).
        """
        if self.state is not None:
            if self.random:
                obs = np.random.uniform(
                    low=0, high=256, size=self.shape).astype(np.uint8)
            else:
                obs = self.state + action
        else:
            raise RuntimeError(
                "DummyEnv: reset() must be called before step()!")
        return obs, 0, True, dict()
