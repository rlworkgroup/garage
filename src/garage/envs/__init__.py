"""Garage wrappers for gym environments."""

from garage.envs.base import GarageEnv
from garage.envs.base import Step
from garage.envs.env_spec import EnvSpec
from garage.envs.grid_world_env import GridWorldEnv
from garage.envs.multi_env_wrapper import (MultiEnvWrapper,
                                           round_robin_strategy,
                                           uniform_random_strategy)
from garage.envs.normalized_env import normalize
from garage.envs.point_env import PointEnv
from garage.envs.task_onehot_wrapper import TaskOnehotWrapper

__all__ = [
    'GarageEnv',
    'Step',
    'EnvSpec',
    'GridWorldEnv',
    'MultiEnvWrapper',
    'normalize',
    'PointEnv',
    'round_robin_strategy',
    'TaskOnehotWrapper',
    'uniform_random_strategy',
]
