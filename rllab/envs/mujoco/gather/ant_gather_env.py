from rllab.envs.mujoco import AntEnv
from rllab.envs.mujoco.gather import GatherEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6
