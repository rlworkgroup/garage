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


def _get_bullet_env_list():
    """Return a complete list of Bullet Gym environments.

    Returns:
        list: a list of bullet environment id (str)
    """
    envs = [env.replace('- ', '') for env in pybullet_envs.getList()]
    # Hardcoded missing environment names from pybullet_envs.getList()
    envs.extend([
        'MinitaurExtendedEnv-v0', 'MinitaurReactiveEnv-v0',
        'MinitaurBallGymEnv-v0', 'MinitaurTrottingEnv-v0',
        'MinitaurStandGymEnv-v0', 'MinitaurAlternatingLegsEnv-v0',
        'MinitaurFourLegStandEnv-v0', 'KukaDiverseObjectGrasping-v0'
    ])
    return envs


def _get_unsupported_env_list():
    """Return a list of unsupported Bullet Gym environments.

    See https://github.com/rlworkgroup/garage/issues/1668

    Returns:
        list: a list of bullet environment id (str)
    """
    return [
        'MinitaurExtendedEnv-v0', 'MinitaurReactiveEnv-v0',
        'MinitaurBallGymEnv-v0', 'MinitaurTrottingEnv-v0',
        'MinitaurStandGymEnv-v0', 'MinitaurAlternatingLegsEnv-v0',
        'MinitaurFourLegStandEnv-v0', 'KukaDiverseObjectGrasping-v0'
    ]
