import gym
import numpy as np

from tests.fixtures.envs.dummy import DummyEnv


class DummyDiscreteEnv(DummyEnv):
    """A dummy discrete environment."""

    def __init__(self, random=True):
        super().__init__(random)

    @property
    def observation_space(self):
        """Return an observation space."""
        return gym.spaces.Box(low=-1, high=1, shape=(1, ), dtype=np.float32)

    @property
    def action_space(self):
        """Return an action space."""
        return gym.spaces.Discrete(2)

    def reset(self):
        """Reset the environment."""
        self.state = np.zeros(1)
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
