import gym
import numpy as np
import theano

from rllab.envs.base import EnvSpec

__all__ = [
    'get_param_values', 'horizon', 'log_diagnostics', 'render',
    'set_param_values', 'spec', 'terminate'
]


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
