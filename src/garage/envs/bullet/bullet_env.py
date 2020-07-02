"""Wrappers for py_bullet environments."""
import inspect

import akro
import gym
from pybullet_envs.bullet.minitaur_duck_gym_env import MinitaurBulletDuckEnv
from pybullet_envs.bullet.minitaur_gym_env import MinitaurBulletEnv
from pybullet_envs.env_bases import MJCFBaseBulletEnv

from garage.envs.env_spec import EnvSpec


class BulletEnv(gym.Wrapper):
    """Binding for py_bullet environments."""

    def __init__(self, env=None, env_name='', is_image=False):
        """Returns a Garage wrapper class for bullet-based gym.Env.

        Args:
            env (gym.wrappers.time_limit): A gym.wrappers.time_limit.TimeLimit
                object wrapping a gym.Env created via gym.make().
            env_name (str): If the env_name is speficied, a gym environment
                with that name will be created. If such an environment does not
                exist, a `gym.error` is thrown.
            is_image (bool): True if observations contain pixel values,
                false otherwise. Setting this to true converts a gym.Spaces.Box
                obs space to an akro.Image and normalizes pixel values.

        """
        if not env:
            # 'RacecarZedBulletEnv-v0' environment enables rendering by
            # default, while pybullet allows only one GUI connection at a time.
            # Setting renders to False avoids potential error when multiple
            # of these envs are tested at the same time.
            if env_name == 'RacecarZedBulletEnv-v0':
                env = gym.make(env_name, renders=False)
            else:
                env = gym.make(env_name)

        # Needed for deserialization
        self._env = env
        self._env_name = env_name

        super().__init__(env)
        self.action_space = akro.from_gym(self.env.action_space)
        self.observation_space = akro.from_gym(self.env.observation_space,
                                               is_image=is_image)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space)

    @property
    def spec(self):
        """Return the environment specification.

        This property needs to exist, since it's defined as a property in
        gym.Wrapper in a way that makes it difficult to overwrite.

        Returns:
            garage.envs.env_spec.EnvSpec: The envionrment specification.

        """
        return self._spec

    def close(self):
        """Close the wrapped env."""
        #  RacecarZedBulletEnv-v0 environment doesn't disconnect from bullet
        #  server in its close() method.
        #  Note that disconnect() disconnects the environment from the physics
        #  server, whereas the GUI window will not be destroyed.
        #  The expected behavior
        if self.env.env.spec.id == 'RacecarZedBulletEnv-v0':
            # pylint: disable=protected-access
            if self.env.env._p.isConnected():
                self.env.env._p.disconnect()
        self.env.close()

    def reset(self, **kwargs):
        """Call reset on wrapped env.

        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Args:
            kwargs: Keyword args

        Returns:
            object: The initial observation.

        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """Call step on wrapped env.

        This method is necessary to suppress a deprecated warning
        thrown by gym.Wrapper.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            np.ndarray: Agent's observation of the current environment
            float: Amount of reward returned after previous action
            bool: Whether the episode has ended, in which case further step()
                calls will return undefined results
            dict: Contains auxiliary diagnostic information (helpful for
                debugging, and sometimes learning)

        """
        observation, reward, done, info = self.env.step(action)
        # gym envs that are wrapped in TimeLimit wrapper modify
        # the done/termination signal to be true whenever a time
        # limit expiration occurs. The following statement sets
        # the done signal to be True only if caused by an
        # environment termination, and not a time limit
        # termination. The time limit termination signal
        # will be saved inside env_infos as
        # 'BulletEnv.TimeLimitTerminated'
        if 'TimeLimit.truncated' in info:
            info['BulletEnv.TimeLimitTerminated'] = done  # done = True always
            done = not info['TimeLimit.truncated']
        return observation, reward, done, info

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

        # Create param name -> param value mapping for the wrapped environment
        args = {key: env.__dict__['_' + key] for key in param_names}

        # Only one local in-process GUI connection is allowed. Thus pickled
        # BulletEnv shouldn't enable rendering. New BulletEnv will connect in
        # DIRECT mode.
        for key in args.keys():
            if 'render' in key:
                args[key] = False

        # Add BulletEnv class specific params
        # env id is saved to help gym.make() in __setstate__
        args['id'] = env.spec.id
        args['env_name'] = self._env_name

        return args

    def __setstate__(self, state):
        """See `Object.__setstate__.

        This will create a new py_bullet client/server connection.

        Args:
            state (dict): The instance’s __init__() arguments.

        """
        env_id = state['id']
        env_name = state['env_name']
        # Create a environment via constructor arguments
        del state['id']
        del state['env_name']
        env = gym.make(env_id, **state)

        self.__init__(env, env_name)
