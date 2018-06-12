from rllab.envs.mujoco import PointEnv
from rllab.envs.mujoco.gather import GatherEnv


class PointGatherEnv(GatherEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2
