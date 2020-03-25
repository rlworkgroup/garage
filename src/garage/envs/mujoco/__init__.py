"""Garage wrappers for mujoco based gym environments."""
try:
    import mujoco_py  # noqa: F401
except Exception as e:
    raise e

from garage.envs.mujoco.half_cheetah_dir_env import HalfCheetahDirEnv
from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv

__all__ = [
    'HalfCheetahDirEnv',
    'HalfCheetahVelEnv',
]
