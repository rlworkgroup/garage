"""Wrappers for the py_bullet based gym environments.

See https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet
"""
try:
    import pybullet_envs  # noqa: F401
except Exception as e:
    raise ImportError('To use garage\'s bullet wrappers, please install '
                      'garage[bullet]')

from garage.envs.bullet.bullet_env import BulletEnv

__all__ = ['BulletEnv']
