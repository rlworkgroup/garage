"""Wrappers for py_bullet environments."""
import inspect
import math

import akro
import gym
from gym.wrappers.time_limit import TimeLimit

from garage import Environment, StepType, TimeStep
from garage.envs.env_spec import EnvSpec

# The bullet gym environments in this list inherit from the
# MJCFBasedBulletEnv class, which doesn't follow a consistent class variable
# naming practice -> constructor param `render` is stored as `isRender`. Thus
# they require additional pickling logic.
MJCFBASED_BULLET_ENV = [
    'ReacherBulletEnv', 'PusherBulletEnv', 'StrikerBulletEnv',
    'ThrowerBulletEnv', 'Walker2DBulletEnv', 'InvertedPendulumBulletEnv',
    'InvertedDoublePendulumBulletEnv', 'InvertedPendulumSwingupBulletEnv',
    'HalfCheetahBulletEnv', 'AntBulletEnv', 'HopperBulletEnv',
    'HumanoidBulletEnv', 'HumanoidFlagrunBulletEnv',
    'HumanoidFlagrunHarderBulletEnv'
]


class BulletEnv(Environment):
    """Binding for py_bullet environments."""

    def __init__(self,
                 env=None,
                 env_name='',
                 is_image=False,
                 max_episode_length=math.inf):
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
            max_episode_length (int): The maximum steps allowed for an episode.

        """
        if not env:
            # 'RacecarZedBulletEnv-v0' environment enables rendering by
            # default, while pybullet allows only one GUI connection at a time.
            # Setting renders to False avoids potential error when multiple
            # of these envs are tested at the same time.
            if env_name.find('RacecarZedBulletEnv') >= 0:
                env = gym.make(env_name, renders=False)
            else:
                env = gym.make(env_name)

        self.env = env

        if isinstance(self.env, TimeLimit):  # env is wrapped by TimeLimit
            self.env._max_episode_steps = max_episode_length
            self._render_modes = self.env.unwrapped.metadata['render.modes']
        elif 'metadata' in env.__dict__:
            self._render_modes = env.metadata['render.modes']
        else:
            self._render_modes = []

        self._last_observation = None
        self._step_cnt = 0
        self._max_episode_length = max_episode_length
        self._visualize = False

        self._action_space = akro.from_gym(self.env.action_space)
        self._observation_space = akro.from_gym(self.env.observation_space,
                                                is_image=is_image)
        self._spec = EnvSpec(action_space=self.action_space,
                             observation_space=self.observation_space,
                             max_episode_length=max_episode_length)

    @property
    def action_space(self):
        """akro.Space: The action space specification."""
        return self._action_space

    @property
    def observation_space(self):
        """akro.Space: The observation space specification."""
        return self._observation_space

    @property
    def spec(self):
        """EnvSpec: The environment specification."""
        return self._spec

    @property
    def render_modes(self):
        """list: A list of string representing the supported render modes."""
        return self._render_modes

    def reset(self, **kwargs):
        """Call reset on wrapped env.

        Args:
            kwargs: Keyword args

        Returns:
            numpy.ndarray: The first observation. It must conforms to
            `observation_space`.
            dict: The episode-level information. Note that this is not part
            of `env_info` provided in `step()`. It contains information of
            the entire episode， which could be needed to determine the first
            action (e.g. in the case of goal-conditioned or MTRL.)

        """
        first_obs = self.env.reset(**kwargs)

        self._step_cnt = 0
        self._last_observation = first_obs
        # Populate episode_info if needed.
        episode_info = {}
        return first_obs, episode_info

    def step(self, action):
        """Call step on wrapped env.

        Args:
            action (np.ndarray): An action provided by the agent.

        Returns:
            TimeStep: The time step resulting from the action.

        Raises:
            RuntimeError: if `step()` is called after the environment has been
                constructed and `reset()` has not been called.

        """
        if self._last_observation is None:
            raise RuntimeError('reset() must be called before step()!')

        observation, reward, done, info = self.env.step(action)

        if self._visualize:
            self.env.render(mode='human')

        last_obs = self._last_observation
        # Type conversion
        if not isinstance(reward, float):
            reward = float(reward)

        self._last_observation = observation
        self._step_cnt += 1

        step_type = None
        if done:
            step_type = StepType.TERMINAL
        elif self._step_cnt == 1:
            step_type = StepType.FIRST
        else:
            step_type = StepType.MID

        # gym envs that are wrapped in TimeLimit wrapper modify
        # the done/termination signal to be true whenever a time
        # limit expiration occurs. The following statement sets
        # the done signal to be True only if caused by an
        # environment termination, and not a time limit
        # termination. The time limit termination signal
        # will be saved inside env_infos as
        # 'BulletEnv.TimeLimitTerminated'
        if 'TimeLimit.truncated' in info or \
            self._step_cnt >= self._spec.max_episode_length:
            info['BulletEnv.TimeLimitTerminated'] = True
            step_type = StepType.TIMEOUT
        else:
            info['TimeLimit.truncated'] = False
            info['BulletEnv.TimeLimitTerminated'] = False

        return TimeStep(
            env_spec=self.spec,
            observation=last_obs,
            action=action,
            reward=reward,
            next_observation=observation,
            env_info=info,
            agent_info={},  # TODO: can't be populated by env
            step_type=step_type)

    def render(self, mode):
        """Renders the environment.

        Args:
            mode (str): the mode to render with. The string must be present in
                `self.render_modes`.
        """
        if mode not in self.render_modes:
            raise ValueError('Supported render modes are {}, but '
                             'got render mode {} instead.'.format(
                                 self.render_modes, mode))
        return self.env.render(mode)

    def visualize(self):
        """Creates a visualization of the environment."""
        self.env.render(mode='human')
        self._visualize = True

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

    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s __init__() arguments

        """
        env = self.env.env

        # Extract constructor signature
        sig = inspect.signature(env.__init__)
        args = {}
        param_names = list(sig.parameters.keys())

        # Hard fix for args/private variable name inconsistency
        if env.spec.id.find('MinitaurBulletEnv') >= 0 or env.spec.id.find(
                'MinitaurBulletDuckEnv') >= 0:
            args['render'] = env._is_render
            param_names.remove('render')
        elif any(env.spec.id.find(id) >= 0 for id in MJCFBASED_BULLET_ENV):
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
        args['max_episode_length'] = self._max_episode_length

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

        self.__init__(env=env, max_episode_length=max_episode_length)
