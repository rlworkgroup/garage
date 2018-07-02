import gym
import numpy as np
import theano

from garage.envs.base import EnvSpec
from garage.misc import ext
from garage.misc import special
from garage.spaces import Box as GarageBox
from garage.spaces import Discrete as GarageDiscrete
from garage.spaces import Product as GarageProduct

__all__ = [
    'bounds', 'default_value', 'flat_dim', 'flatten', 'flatten_n', 'sample',
    'spec', 'unflatten', 'unflatten_n', 'weighted_sample',
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
        return np.sum([flat_dim(x) for x in space.spaces])
    else:
        raise NotImplementedError


def flatten(space, obs):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).flatten()
    elif isinstance(space, gym.spaces.Discrete):
        if space.n == 2:
            obs = int(obs)
        return special.to_onehot(obs, space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return np.concatenate(
            [flatten(c, xi) for c, xi in zip(space.spaces, obs)])
    else:
        raise NotImplementedError


def flatten_n(space, obs):
    if isinstance(space, gym.spaces.Box):
        obs = np.asarray(obs)
        return obs.reshape((obs.shape[0], -1))
    elif isinstance(space, gym.spaces.Discrete):
        return special.to_onehot_n(obs, space.n)
    elif isinstance(space, gym.spaces.Tuple):
        obs_regrouped = [[obs[i] for o in obs] for i in range(len(obs[0]))]
        flat_regrouped = [
            flatten_n(c, oi) for c, oi in zip(space.spaces, obs_regrouped)
        ]
        return np.concatenate(flat_regrouped, axis=-1)
    else:
        raise NotImplementedError


def horizon(env):
    return env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']


def new_tensor_variable(space, name, extra_dims):
    if isinstance(space, gym.spaces.Box):
        return ext.new_tensor(
            name=name, ndim=extra_dims + 1, dtype=theano.config.floatX)
    elif isinstance(space, gym.spaces.Discrete):
        if space.n <= 2**8:
            return ext.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint8')
        elif space.n <= 2**16:
            return ext.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint16')
        else:
            return ext.new_tensor(
                name=name, ndim=extra_dims + 1, dtype='uint32')
    elif isinstance(space, gym.spaces.Tuple):
        dtypes = [
            new_tensor_variable(c, "tmp", extra_dims=0).dtype
            for c in space.spaces
        ]
        if dtypes and hasattr(dtypes[0], "as_numpy_dtype"):
            dtypes = [d.as_numpy_dtype for d in dtypes]
        common_dtype = np.core.numerictypes.find_common_type([], dtypes)
        return ext.new_tensor(
            name=name,
            ndim=extra_dims + 1,
            dtype=common_dtype,
        )
    else:
        raise NotImplementedError


def sample(space):
    if isinstance(space, gym.spaces.Tuple):
        return tuple(x.sample() for x in space.spaces)
    else:
        raise NotImplementedError


def spec(env):
    return EnvSpec(
        observation_space=env.observation_space,
        action_space=env.action_space,
    )


def _to_garage_space(space):
    if isinstance(space, gym.spaces.Box):
        return GarageBox(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return GarageDiscrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return GarageProduct([_to_garage_space(s) for s in space.spaces])
    else:
        raise NotImplementedError


def unflatten(space, obs):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).reshape(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return special.from_onehot(obs)
    elif isinstance(space, gym.spaces.Tuple):
        dims = [flat_dim(c) for c in space.spaces]
        flat_xs = np.split(obs, np.cumsum(dims)[:-1])
        return tuple(unflatten(c, xi) for c, xi in zip(space.spaces, flat_xs))
    else:
        raise NotImplementedError


def unflatten_n(space, obs):
    if isinstance(space, gym.spaces.Box):
        obs = np.asarray(obs)
        return obs.reshape((obs.shape[0], ) + space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return special.from_onehot_n(obs)
    elif isinstance(space, gym.spaces.Tuple):
        dims = [flat_dim(c) for c in space.spaces]
        flat_xs = np.split(obs, np.cumsum(dims)[:-1], axis=-1)
        unflat_xs = [
            unflatten_n(c, xi) for c, xi in zip(space.spaces, flat_xs)
        ]
        unflat_xs_grouped = list(zip(*unflat_xs))
        return unflat_xs_grouped
    else:
        raise NotImplementedError


def weighted_sample(space, weights):
    if isinstance(space, gym.spaces.Discrete):
        return special.weighted_sample(weights, range(space.n))
    else:
        raise NotImplementedError
