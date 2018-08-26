"""
Wrappers for the DeepMind Control Suite.

See https://github.com/deepmind/dm_control
"""
from garage.envs.dm_control.dm_control_viewer import DmControlViewer
from garage.envs.dm_control.dm_control_env import DmControlEnv  # noqa: I100

__all__ = ["DmControlViewer", "DmControlEnv"]
