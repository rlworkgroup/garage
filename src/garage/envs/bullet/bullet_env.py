"""Wrappers for py_bullet environments."""
import inspect

import gym

from garage.envs import GymEnv

# The bullet gym environments in this list inherit from the
# MJCFBasedBulletEnv class, which doesn't follow a consistent class variable
# naming practice -> constructor param `render` is stored as `isRender`. Thus
# they require additional pickling logic.
_MJCF_BASED_BULLET_ENVS = [
    'ReacherBulletEnv', 'PusherBulletEnv', 'StrikerBulletEnv',
    'ThrowerBulletEnv', 'Walker2DBulletEnv', 'InvertedPendulumBulletEnv',
    'InvertedDoublePendulumBulletEnv', 'InvertedPendulumSwingupBulletEnv',
    'HalfCheetahBulletEnv', 'AntBulletEnv', 'HopperBulletEnv',
    'HumanoidBulletEnv', 'HumanoidFlagrunBulletEnv',
    'HumanoidFlagrunHarderBulletEnv'
]


class BulletEnv(GymEnv):
    """Binding for py_bullet environments."""

    def __new__(cls, *args, **kwargs):
        """Overwrite GymEnv's __new__ logic.

        Args:
            *args: Positional arguments to :class:`GymEnv`
            **kwargs: Keyword arguments to :class:`GymEnv`

        Returns:
             BulletEnv: the wrapped bullet environment.

        """
        return super(BulletEnv, cls).__new__(cls)

    def __init__(self, env, is_image=False, max_episode_length=None):
        """Returns a wrapper class for bullet-based gym.Env.

        Args:
            env (gym.wrappers.time_limit or str): A gym.TimeLimit
                object wrapping a gym.Env created via gym.make(). Or a name
                of the gym environment to be created.
            is_image (bool): True if observations contain pixel values,
                false otherwise. Setting this to true converts a gym.Spaces.Box
                obs space to an akro.Image and normalizes pixel values.
            max_episode_length (int): The maximum steps allowed for an episode.

        """
        env_name = None
        if isinstance(env, str):
            env_name = env
        elif isinstance(env, gym.Env):
            env_name = env.spec.id

        # 'RacecarZedBulletEnv-v0' environment enables rendering by
        # default, while pybullet allows only one GUI connection at a time.
        # Setting renders to False avoids potential error when multiple
        # of these envs are tested at the same time.
        if 'RacecarZedBulletEnv' in env_name:
            env = gym.make(env_name, renders=False)

        super().__init__(env,
                         is_image=is_image,
                         max_episode_length=max_episode_length)

    def close(self):
        """Close the wrapped env."""
        #  RacecarZedBulletEnv-v0 environment doesn't disconnect from bullet
        #  server in its close() method.
        #  Note that disconnect() disconnects the environment from the physics
        #  server, whereas the GUI window will not be destroyed.
        #  The expected behavior
        if 'RacecarZedBulletEnv' in self._env.env.spec.id:
            # pylint: disable=protected-access
            if self._env.env._p.isConnected():
                self._env.env._p.disconnect()
        self._env.close()

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
        if ('MinitaurBulletEnv' in env.spec.id
                or 'MinitaurBulletDuckEnv' in env.spec.id):
            args['render'] = env._is_render
            param_names.remove('render')
        elif any(id in env.spec.id for id in _MJCF_BASED_BULLET_ENVS):
            args['render'] = env.isRender
            if 'render' in param_names:
                param_names.remove('render')
            if 'robot' in param_names:
                args['robot'] = env.robot
                param_names.remove('robot')

        # Create param name -> param value mapping for the wrapped environment
        args = {
            key: env.__dict__['_' + key]
            for key in param_names if '_' + key in env.__dict__
        }

        # Only one local in-process GUI connection is allowed. Thus pickled
        # BulletEnv shouldn't enable rendering. New BulletEnv will connect in
        # DIRECT mode.
        for key in args.keys():
            if 'render' in key:
                args[key] = False

        # Add BulletEnv class specific params
        # env id is saved to help gym.make() in __setstate__
        args['id'] = env.spec.id
        args['max_episode_length'] = self._max_episode_length

        if 'kwargs' in args:
            del args['kwargs']

        return args

    def __setstate__(self, state):
        """See `Object.__setstate__.

        This will create a new py_bullet client/server connection.

        Args:
            state (dict): The instance’s __init__() arguments.

        """
        env_id = state['id']
        max_episode_length = state['max_episode_length']
        # Create a environment via constructor arguments
        del state['id']
        del state['max_episode_length']
        env = gym.make(env_id, **state)

        self.__init__(env, max_episode_length=max_episode_length)
