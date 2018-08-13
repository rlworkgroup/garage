import gym
import numpy as np

from garage.misc import special
from garage.spaces import Space

__all__ = [
    'configure_dims', 'dims_to_shapes', 'flat_dim', 'flatten', 'flatten_n',
    'unflatten', 'unflatten_n', 'weighted_sample'
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


def dims_to_shapes(input_dims):
    return {
        key: tuple([val]) if val > 0 else tuple()
        for key, val in input_dims.items()
    }


def configure_dims(env):
    env.reset()
    _, _, _, info = env.step(env.action_space.sample())
    obs = env.observation_space
    action = env.action_space

    if isinstance(obs, gym.spaces.Dict):
        dims = {
            "observation": flat_dim(obs.spaces["observation"]),
            "action": flat_dim(action),
            "goal": flat_dim(obs.spaces["desired_goal"]),
            "achieved_goal": flat_dim(obs.spaces["achieved_goal"]),
        }

        for key, value in info.items():
            value = np.array(value)
            if value.ndim == 0:
                value = value.reshape(1)
            dims['info_{}'.format(key)] = value.shape[0]
    else:
        if isinstance(obs, Space):
            dims = {
                "observation": obs.flat_dim,
                "action": action.flat_dim,
                "terminal": 1,
                "reward": 1,
                "next_observation": obs.flat_dim,
            }
        else:
            dims = {
                "observation": flat_dim(obs),
                "action": flat_dim(action),
                "terminal": 1,
                "reward": 1,
                "next_observation": flat_dim(obs),
            }

    return dims
