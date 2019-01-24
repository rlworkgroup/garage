from collections import namedtuple

import numpy as np

EnvSpec = namedtuple('EnvSpec', ['action_space', 'observation_space'])
Space = namedtuple('Space', 'flat_dim')


def _get_flat_dim(space):
    return np.prod(space.low.shape)


def get_env_spec(env):
    action_space = env.action_space
    obs_space = env.observation_space
    spec = EnvSpec(Space(_get_flat_dim(action_space)), Space(_get_flat_dim(obs_space)))
    return spec
