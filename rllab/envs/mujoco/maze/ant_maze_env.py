from rllab.envs.mujoco import AntEnv
from rllab.envs.mujoco.maze import MazeEnv


class AntMazeEnv(MazeEnv):

    MODEL_CLASS = AntEnv
    ORI_IND = 6

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0
