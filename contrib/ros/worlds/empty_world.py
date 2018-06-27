"""world without any objects."""

import collections

import gym
import numpy as np

from contrib.ros.worlds.world import World


class EmptyWorld(World):
    """Empty world class."""

    def __init__(self, simulated=False):
        """
        Users use this to manage world and get world state.

        :param simulated: Bool
                if simulated
        """
        self._simulated = simulated

    def initialize(self):
        """Use this to initialize the world."""
        pass

    def reset(self):
        """Use this to reset the world."""
        pass

    def terminate(self):
        """Use this to terminate the world."""
        pass

    def get_observation(self):
        """Use this to get the observation from world."""
        achieved_goal = np.array([])

        obs = np.array([])

        Observation = collections.namedtuple('Observation',
                                             'obs achieved_goal')

        observation = Observation(obs=obs, achieved_goal=achieved_goal)

        return observation

    @property
    def observation_space(self):
        """Use this to get observation space."""
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.get_observation().obs.shape,
            dtype=np.float32)
