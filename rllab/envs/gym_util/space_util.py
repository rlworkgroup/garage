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


def components(space):
    if isinstance(space, gym.spaces.Tuple):
        return self.spaces
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
        return special.to_onehot(obs, space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return np.concatenate(
            [flatten(xi, c) for c, xi in zip(space.spaces, obs)])
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


def log_diagnostics(space, paths):
    pass


def sample(space):
    if isinstance(space, gym.spaces.Tuple):
        return tuple(x.sample() for x in self.spaces)
    else:
        raise NotImplementedError


def unflatten(space, obs):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).reshape(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return special.from_onehot(obs)
    elif isinstance(space, gym.spaces.Tuple):
        dims = [flat_dim(c) for c in space.spaces]
        flat_xs = np.split(x, np.cumsum(dims)[:-1])
        return tuple(unflatten(c, xi) for c, xi in zip(self.spaces, flat_xs))
    else:
        raise NotImplementedError


def unflatten_n(space, obs):
    if isinstance(space, gym.spaces.Box):
        obs = np.asarray(obs)
        return obs.reshape((obs.shape[0], ) + space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return special.from_onehot_n(obs)
    elif isinstance(space, gym.spaces.Tuple):
        dims = [flat_dim(c) for c in self.spaces]
        flat_xs = np.split(xs, np.cumsum(dims)[:-1], axis=-1)
        unflat_xs = [unflatten_n(c, xi) for c, xi in zip(self.spaces, flat_xs)]
        unflat_xs_grouped = list(zip(*unflat_xs))
        return unflat_xs_grouped
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
    elif isinstance(space, gym.spaces.Tuple):
        dtypes = [
            new_tensor_variable(c, "tmp", extra_dims=0).dtype
            for c in space.spaces
        ]
        if len(dtypes) > 0 and hasattr(dtypes[0], "as_numpy_dtype"):
            dtypes = [d.as_numpy_dtype for d in dtypes]
        common_dtype = np.core.numerictypes.find_common_type([], dtypes)
        return ext.new_tensor(
            name=name,
            ndim=extra_dims + 1,
            dtype=common_dtype,
        )
    else:
        raise NotImplementedError
