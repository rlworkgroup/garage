import numpy as np
import rospy

from rllab.core.serializable import Serializable
from rllab.envs.base import Env
from rllab.misc.ext import get_seed


class RosEnv(Env, Serializable):
    """
    Superclass for all ros environment
    """

    def __init__(self):
        Serializable.quick_init(self, locals())

        np.random.RandomState(get_seed())

    def initialize(self):
        # TODO (gh/74: Add initialize interface for robot)
        pass

    # =======================================================
    # The functions that base rllab Env asks to implement
    # =======================================================
    def step(self, action):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def observation_space(self):
        raise NotImplementedError

    # ====================================================
    # Need to be implemented in specific robot env
    # ====================================================
    def sample_goal(self):
        """
        Samples a new goal and returns it.
        """
        raise NotImplementedError

    @property
    def goal(self):
        return self._goal

    @goal.setter
    def goal(self, value):
        self._goal = value
