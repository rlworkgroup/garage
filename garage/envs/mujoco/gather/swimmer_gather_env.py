from garage.envs.mujoco import SwimmerEnv
from garage.envs.mujoco.gather import GatherEnv


class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2
