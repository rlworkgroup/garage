import numpy as np

from rllab.envs.mujoco import Walker2DEnv
from rllab.envs.mujoco.hill import HillEnv
from rllab.envs.mujoco.hill import terrain
from rllab.misc.overrides import overrides
from rllab.spaces import Box


class Walker2DHillEnv(HillEnv):

    MODEL_CLASS = Walker2DEnv

    @overrides
    def _mod_hfield(self, hfield):
        # clear a flat patch for the robot to start off from
        return terrain.clear_patch(
            hfield, Box(np.array([-2.0, -2.0]), np.array([-0.5, -0.5])))
