from garage.envs.mujoco import PointEnv
from garage.envs.mujoco.maze import MazeEnv


class PointMazeEnv(MazeEnv):

    MODEL_CLASS = PointEnv
    ORI_IND = 2

    MAZE_HEIGHT = 2
    MAZE_SIZE_SCALING = 3.0

    MANUAL_COLLISION = True
