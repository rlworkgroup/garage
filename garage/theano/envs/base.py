"""Wrapper class that converts gym.Env into TheanoEnv."""
from cached_property import cached_property
from gym.spaces import Box as GymBox
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple

from garage.envs import EnvSpec
from garage.envs import ProxyEnv
from garage.theano.spaces import Box
from garage.theano.spaces import Discrete
from garage.theano.spaces import Product


def to_theano_space(space):
    """Convert gym.space into theano.space."""
    if isinstance(space, GymBox):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, GymDiscrete):
        return Discrete(space.n)
    elif isinstance(space, GymTuple):
        return Product(list(map(to_theano_space, space.spaces)))
    else:
        raise NotImplementedError


class TheanoEnv(ProxyEnv):
    """Return a Theano wrapper class for gym.Env."""

    @cached_property
    def observation_space(self):
        """Convert gym.observation_space into theano.space."""
        return to_theano_space(self.wrapped_env.observation_space)

    @cached_property
    def action_space(self):
        """Convert gym.action_space into theano.space."""
        return to_theano_space(self.wrapped_env.action_space)

    @cached_property
    def spec(self):
        """Return an EnvSpec."""
        return EnvSpec(
            observation_space=self.observation_space,
            action_space=self.action_space,
        )

    @cached_property
    def horizon(self):
        """Return spec tags."""
        return self.wrapped_env.spec.tags[
            'wrapper_config.TimeLimit.max_episode_steps']
