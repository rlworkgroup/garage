"""Garage wrappers for gym environments."""

from garage.envs.grid_world_env import GridWorldEnv
from garage.envs.gym_env import GymEnv
from garage.envs.multi_env_wrapper import MultiEnvWrapper
from garage.envs.normalized_env import normalize
from garage.envs.point_env import PointEnv
from garage.envs.task_onehot_wrapper import TaskOnehotWrapper

__all__ = [
    'GymEnv',
    'GridWorldEnv',
    'MultiEnvWrapper',
    'normalize',
    'PointEnv',
    'TaskOnehotWrapper',
]
