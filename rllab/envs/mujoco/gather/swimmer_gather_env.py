from rllab.envs.mujoco import SwimmerEnv
from rllab.envs.mujoco.gather import GatherEnv


class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2
