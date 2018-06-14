from garage.envs.mujoco import AntEnv
from garage.envs.mujoco.gather import GatherEnv


class AntGatherEnv(GatherEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6
