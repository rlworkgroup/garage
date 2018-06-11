from rllab.envs.mujoco import GatherEnv
from rllab.envs.mujoco import PointEnv


class PointGatherEnv(GatherEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2
