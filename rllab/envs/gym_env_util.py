import gym
import numpy as np
import theano

from rllab.envs.base import EnvSpec
from rllab.envs.gym_space_util import flat_dim
from rllab.misc import ext
from rllab.misc import special
from rllab.misc.overrides import overrides

__all__ = [
    'action_dim', 'get_param_values', 'horizon', 'log_diagnostics', 'render',
    'set_param_values', 'spec', 'terminate'
]


def action_dim(env):
    return flat_dim(env.action_space)


def get_param_values(env):
    return None


def horizon(env):
    raise NotImplementedError


def render(env):
    pass


def set_param_values(env, params):
    pass


def spec(env):
    return EnvSpec(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )


def terminate(env):
    pass
