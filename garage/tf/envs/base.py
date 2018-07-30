from cached_property import cached_property

from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple

from garage.envs import GarageEnv
from garage.tf.spaces import Box
from garage.tf.spaces import Discrete
from garage.tf.spaces import Product
from garage.misc.overrides import overrides


class TfEnv(GarageEnv):
    """
    Returns a TensorFlow wrapper class for gym.Env.

    Args:
        env (gym.Env): the env that will be wrapped
    """

    def __init__(self, env):
        super().__init__(env)

    @classmethod
    def wrap(cls, env_cls, **extra_kwargs):
        # Use a class wrapper rather than a lambda method for smoother
        # serialization
        return WrappedCls(cls, env_cls, extra_kwargs)

    @overrides
    def _to_garage_space(self, space):
        """
        Converts gym.space to a TensorFlow space.

        Returns:
            space (garage.tf.spaces)
        """
        if isinstance(space, GymBox):
            return Box(low=space.low, high=space.high)
        elif isinstance(space, GymDiscrete):
            return Discrete(space.n)
        elif isinstance(space, GymTuple):
            return Product(list(map(self._to_tf_space, space.spaces)))
        else:
            raise NotImplementedError

    @cached_property
    @overrides
    def action_space(self):
        """Returns a converted action_space."""
        return self._to_garage_space(self.action_space)

    @cached_property
    @overrides
    def observation_space(self):
        """Returns a converted observation_space."""
        return self._to_garage_space(self.observation_space)


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
