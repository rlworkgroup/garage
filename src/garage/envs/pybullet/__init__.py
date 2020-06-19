"""Wrappers for the Pybullet based gym environments.

See https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet
"""
try:
    import pybullet_envs  # noqa: F401
except Exception as e:
    raise ImportError("To use garage's pybullet wrappers, please install "
                      'garage[pybullet].')

from garage.envs.pybullet.pybullet_env import PybulletEnv

__all__ = ['PybulletEnv']
