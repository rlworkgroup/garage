import collections

import gym
import numpy as np

from contrib.ros.worlds.world import World


class EmptyWorld(World):
    def __init__(self, simulated=False):
        """
        Users use this to manage world and get world state.
        """
        self._simulated = simulated

    def initialize(self):
        pass

    def reset(self):
        pass

    def terminate(self):
        pass

    def get_observation(self):
        achieved_goal = np.array([])

        obs = np.array([])

        Observation = collections.namedtuple('Observation',
                                             'obs achieved_goal')

        observation = Observation(obs=obs, achieved_goal=achieved_goal)

        return observation

    @property
    def observation_space(self):
        return gym.spaces.Box(
            -np.inf,
            np.inf,
            shape=self.get_observation().obs.shape,
            dtype=np.float32)
