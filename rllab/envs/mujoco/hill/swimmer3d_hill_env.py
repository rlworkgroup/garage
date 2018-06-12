import gym
import numpy as np

from rllab.envs.mujoco import Swimmer3DEnv
from rllab.envs.mujoco.hill import HillEnv
from rllab.envs.mujoco.hill import terrain
from rllab.misc.overrides import overrides


class Swimmer3DHillEnv(HillEnv):

    MODEL_CLASS = Swimmer3DEnv

    @overrides
    def _mod_hfield(self, hfield):
        # clear a flat patch for the robot to start off from
        return terrain.clear_patch(
            hfield,
            gym.spaces.Box(
                np.array([-3.0, -1.5]),
                np.array([0.0, -0.5]),
                dtype=np.float32))
