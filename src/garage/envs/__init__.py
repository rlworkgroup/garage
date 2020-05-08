"""Garage wrappers for gym environments."""

from garage.envs.env_spec import EnvSpec
from garage.envs.garage_env import GarageEnv
from garage.envs.grid_world_env import GridWorldEnv
from garage.envs.multi_env_wrapper import MultiEnvWrapper
from garage.envs.normalized_env import normalize
from garage.envs.point_env import PointEnv
from garage.envs.step import Step
from garage.envs.task_onehot_wrapper import TaskOnehotWrapper

__all__ = [
    'GarageEnv',
    'Step',
    'EnvSpec',
    'GridWorldEnv',
    'MultiEnvWrapper',
    'normalize',
    'PointEnv',
    'TaskOnehotWrapper',
]
