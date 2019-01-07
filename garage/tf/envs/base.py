from cached_property import cached_property
from gym.spaces import Box as GymBox
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple

from garage.envs import GarageEnv
from garage.envs.env_spec import EnvSpec
from garage.misc.overrides import overrides
from garage.tf.spaces import Box
from garage.tf.spaces import Dict
from garage.tf.spaces import Discrete
from garage.tf.spaces import Tuple


class TfEnv(GarageEnv):
    """
    Returns a TensorFlow wrapper class for gym.Env.

    Args:
        env (gym.Env): the env that will be wrapped
    """

    def __init__(self, env=None, env_name=""):
        super().__init__(env, env_name)
        self.action_space = self._to_garage_space(self.env.action_space)
        self.observation_space = self._to_garage_space(
            self.env.observation_space)

    @classmethod
    def wrap(cls, env_cls, **extra_kwargs):
        # Use a class wrapper rather than a lambda method for smoother
        # serialization
        return WrappedCls(cls, env_cls, extra_kwargs)

    @overrides
    def _to_garage_space(self, space):
        """
        Converts a gym.space to a garage.tf.space.

        Returns:
            space (garage.tf.spaces)
        """
        if isinstance(space, GymBox):
            return Box(low=space.low, high=space.high, dtype=space.dtype)
        elif isinstance(space, GymDict):
            return Dict(space.spaces)
        elif isinstance(space, GymDiscrete):
            return Discrete(space.n)
        elif isinstance(space, GymTuple):
            return Tuple(list(map(self._to_garage_space, space.spaces)))
        else:
            raise NotImplementedError

    @cached_property
    @overrides
    def spec(self):
        """
        Returns an EnvSpec.

        Returns:
            spec (garage.envs.EnvSpec)
        """
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space)

    @cached_property
    @overrides
    def max_episode_steps(self):
        """
        Returns gym.Env's max episode steps.
         Returns:
            max_episode_steps (int)
        """
        return self.env.spec.max_episode_steps


class VecTfEnv:
    def __init__(self, vec_env):
        self.vec_env = vec_env

    def reset(self):
        return self.vec_env.reset()

    @property
    def num_envs(self):
        return self.vec_env.num_envs

    def step(self, action_n):
        return self.vec_env.step(action_n)

    def close(self):
        self.vec_env.close()


class WrappedCls:
    def __init__(self, cls, env_cls, extra_kwargs):
        self.cls = cls
        self.env_cls = env_cls
        self.extra_kwargs = extra_kwargs

    def __call__(self, *args, **kwargs):
        return self.cls(
            self.env_cls(*args, **dict(self.extra_kwargs, **kwargs)))
