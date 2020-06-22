"""Wrappers for py_bullet environments."""
import inspect

from pybullet_envs.bullet.minitaur_duck_gym_env import MinitaurBulletDuckEnv
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
from pybullet_envs.env_bases import MJCFBaseBulletEnv

from garage.envs import GarageEnv


class BulletEnv(GarageEnv):
    """Binding for py_bullet environments."""

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s __init__() arguments

        """
        env = self._env.env

        # Extract constructor signature
        sig = inspect.signature(env.__init__)
        args = {}
        param_names = list(sig.parameters.keys())

        # Hard fix for args/private variable name inconsistency
        if isinstance(env, (MinitaurBulletEnv, MinitaurBulletDuckEnv)):
            args['render'] = env._is_render
            param_names.remove('render')
        elif issubclass(type(env), MJCFBaseBulletEnv):
            args['render'] = env.isRender
            if 'render' in param_names:
                param_names.remove('render')
            if 'robot' in param_names:
                args['robot'] = env.robot
                param_names.remove('robot')

        # Create param name -> param value mapping
        args = {key: env.__dict__['_' + key] for key in param_names}
        args['class_type'] = type(env)
        args['_env_name'] = self._env_name

        return args

    def __setstate__(self, state):
        """See `Object.__setstate__.

        This will create a new py_bullet client/server connection.

        Args:
            state (dict): The instance’s __init__() arguments.

        """
        class_type = state['class_type']
        env_name = state['_env_name']
        # Create a new class instance via constructor arguments
        del state['class_type']
        del state['_env_name']
        env = class_type(**state)

        self.__init__(env, env_name)
