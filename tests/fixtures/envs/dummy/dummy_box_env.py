"""Dummy gym.spaces.Box environment for testing purpose."""
import gym
import numpy as np

from tests.fixtures.envs.dummy import DummyEnv


class DummyBoxEnv(DummyEnv):
    """A dummy gym.spaces.Box environment.

    Args:
        random (bool): If observations are randomly generated or not.
        obs_dim (iterable): Observation space dimension.
        action_dim (iterable): Action space dimension.

    """

    def __init__(self, random=True, obs_dim=(4, ), action_dim=(2, )):
        super().__init__(random, obs_dim, action_dim)

    @property
    def observation_space(self):
        """Return an observation space.

        Returns:
            gym.spaces: The observation space of the environment.

        """
        return gym.spaces.Box(low=-1,
                              high=1,
                              shape=self._obs_dim,
                              dtype=np.float32)

    @property
    def action_space(self):
        """Return an action space.

        Returns:
            gym.spaces: The action space of the environment.

        """
        return gym.spaces.Box(low=-5.0,
                              high=5.0,
                              shape=self._action_dim,
                              dtype=np.float32)

    def reset(self):
        """Reset the environment.

        Returns:
            np.ndarray: The observation obtained after reset.

        """
        return np.ones(self._obs_dim, dtype=np.float32)

    def step(self, action):
        """Step the environment.

        Args:
            action (int): Action input.

        Returns:
            np.ndarray: Observation.
            float: Reward.
            bool: If the environment is terminated.
            dict: Environment information.

        """
        return self.observation_space.sample(), 0, False, dict(dummy='dummy')
