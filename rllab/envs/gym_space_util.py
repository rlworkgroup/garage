import gym
import numpy as np
import theano

from rllab.misc import ext
from rllab.misc import special
from rllab.misc.overrides import overrides

__all__ = [
    'bounds', 'default_value', 'flatten_dim', 'flatten', 'flatten_n',
    'log_diagnostics', 'unflatten', 'unflatten_n', 'weighted_sample',
    'new_tensor_variable'
]


def bounds(space):
    if isinstance(space, gym.spaces.Box):
        return space.low, space.high
    else:
        raise NotImplementedError


def default_value(space):
    if isinstance(space, gym.spaces.Discrete):
        return 0
    else:
        return NotImplementedError


def flat_dim(space):
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.low.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Tuple):
        return np.sum([gym_space_flat_dim(x) for x in space.spaces])
    else:
        raise NotImplementedError


def flatten(space, obs):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).flatten()
    elif isinstance(space, gym.spaces.Discrete):
        return special.to_onehot(obs, space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return np.concatenate(
            [gym_space_flatten(xi, c) for c, xi in zip(space.spaces, obs)])
    else:
        raise NotImplementedError


def flatten_n(space, obs):
    if isinstance(space, gym.spaces.Box):
        obs = np.asarray(obs)
        return obs.reshape((obs.shape[0], -1))
    elif isinstance(space, gym.spaces.Discrete):
        return special.to_onehot_n(obs, space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return np.concatenate(
            [gym_space_flatten_n(xi, c) for c, xi in zip(space.spaces, obs)])
    else:
        raise NotImplementedError


def log_diagnostics(space, paths):
    pass


def unflatten(space, obs):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).reshape(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return special.from_onehot(obs)
    elif isinstance(space, gym.spaces.Tuple):
        return np.concatenate(
            [gym_space_unflatten(xi, c) for c, xi in zip(space.spaces, obs)])
    else:
        raise NotImplementedError


def unflatten_n(space, obs):
    if isinstance(space, gym.spaces.Box):
        obs = np.asarray(obs)
        return obs.reshape((obs.shape[0], ) + space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return special.from_onehot_n(obs)
    elif isinstance(space, gym.spaces.Tuple):
        return np.concatenate(
            [gym_space_unflatten_n(xi, c) for c, xi in zip(space.spaces, obs)])
    else:
        raise NotImplementedError


def weighted_sample(space, obs):
    if isinstance(space, gym.spaces.Discrete):
        return special.weighted_sample(weights, range(space.n))
    else:
        raise NotImplementedError


def new_tensor_variable(space, name, extra_dims):
    if isinstance(space, gym.spaces.Box):
        return ext.new_tensor(
            name=name, ndim=extra_dims + 1, dtype=theano.config.floatX)
    elif isinstance(space, gym.spaces.Discrete):
        if self.n <= 2**8:
            return ext.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint8')
        elif self.n <= 2**16:
            return ext.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint16')
        else:
            return ext.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint32')
    else:
        raise NotImplementedError
