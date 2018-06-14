from garage.envs.mujoco import PointEnv
from garage.envs.mujoco.gather import GatherEnv


class PointGatherEnv(GatherEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2
