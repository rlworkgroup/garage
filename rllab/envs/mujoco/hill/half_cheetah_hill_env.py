import gym
import numpy as np

from rllab.envs.mujoco import HalfCheetahEnv
from rllab.envs.mujoco.hill import HillEnv
from rllab.envs.mujoco.hill import terrain
from rllab.misc.overrides import overrides


class HalfCheetahHillEnv(HillEnv):

    MODEL_CLASS = HalfCheetahEnv

    @overrides
    def _mod_hfield(self, hfield):
        # clear a flat patch for the robot to start off from
        return terrain.clear_patch(
            hfield,
            gym.spaces.Box(
                np.array([-3.0, -1.5]),
                np.array([0.0, -0.5]),
                dtype=np.float32))
