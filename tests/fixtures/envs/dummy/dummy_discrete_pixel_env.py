from unittest import mock

import gym
import numpy as np

from tests.fixtures.envs.dummy import DummyEnv


class DummyDiscretePixelEnv(DummyEnv):
    """
    A dummy discrete pixel environment.

    It follows Atari game convention, where actions are 'NOOP', 'FIRE', ...
    It also contains self.unwrapped.ale.lives, get_action_meanings for testing.
    Observations are as following:
        after reset    : np.zeros(self._shape).
        action 1 (FIRE): np.ones(self._shape).
        otherwise      : random if self.random is True,
            otherwise previous state + action.
    """

    def __init__(self, random=True):
        super().__init__(random)
        self.unwrapped.get_action_meanings = self._get_action_meanings
        self.unwrapped.ale = mock.Mock()
        self.unwrapped.ale.lives = self.get_lives
        self._shape = (10, 10, 3)
        self._observation_space = gym.spaces.Box(
            low=0, high=255, shape=self._shape, dtype=np.uint8)
        self.step_called = 0

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
        return gym.spaces.Discrete(5)

    def _get_action_meanings(self):
        return ['NOOP', 'FIRE', 'SLEEP', 'EAT', 'PLAY']

    def get_lives(self):
        """Get number of lives."""
        return self._lives

    def reset(self):
        """Reset the environment."""
        self.state = np.zeros(self._shape, dtype=np.uint8)
        self._lives = 5
        self.step_called = 0
        return self.state

    def step(self, action):
        """
        Step the environment.

        Before gym fixed overflow issue for sample() in
        np.uint8 environment, we will handle the sampling here.
        We need high=256 since np.random.uniform sample from [low, high)
        (includes low, but excludes high).
        """
        done = False
        if self.state is not None:
            # Simulating FIRE action
            if action == 1:
                obs = np.ones(self._shape, dtype=np.uint8)
            else:
                if self.random:
                    obs = np.random.uniform(
                        low=0, high=256, size=self._shape).astype(np.uint8)
                else:
                    obs = self.state + action
            if self._lives == 0:
                raise RuntimeError("DummyEnv: Cannot step when lives = 0!")
            self._lives -= 1
            if self._lives == 0:
                done = True
        else:
            raise RuntimeError(
                "DummyEnv: reset() must be called before step()!")
        self.step_called += 1

        return obs, 0, done, {'ale.lives': self._lives}
