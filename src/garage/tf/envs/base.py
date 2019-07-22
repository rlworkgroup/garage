import akro
from cached_property import cached_property

from garage.envs import GarageEnv
from garage.misc.overrides import overrides


class TfEnv(GarageEnv):
    """
    Returns a TensorFlow wrapper class for gym.Env.

    Args:
        env (gym.Env): the env that will be wrapped
    """
    def __init__(self, env=None, env_name=''):
        super().__init__(env, env_name)
        self.action_space = akro.from_gym(self.env.action_space)
        self.observation_space = akro.from_gym(self.env.observation_space)

    @classmethod
    def wrap(cls, env_cls, **extra_kwargs):
        # Use a class wrapper rather than a lambda method for smoother
        # serialization
        return WrappedCls(cls, env_cls, extra_kwargs)

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
