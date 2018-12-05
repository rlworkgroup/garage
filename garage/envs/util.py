import gym.spaces
import numpy as np

from garage.misc import special

__all__ = [
    'flat_dim', 'flatten', 'flatten_n', 'unflatten', 'unflatten_n',
    'weighted_sample'
]


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
