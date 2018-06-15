import gym
import numpy as np

from garage.core import Serializable
from garage.misc import special
from garage.misc.overrides import overrides


def gym_space_flatten_dim(space):
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.low.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Tuple):
        return np.sum([gym_space_flatten_dim(x) for x in space.spaces])
    else:
        raise NotImplementedError


def gym_space_flatten(space, obs):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).flatten()
    elif isinstance(space, gym.spaces.Discrete):
        if space.n == 2:
            obs = int(obs)
        return special.to_onehot(obs, space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return np.concatenate(
            [gym_space_flatten(c, xi) for c, xi in zip(space.spaces, obs)])
    else:
        raise NotImplementedError


def gym_space_unflatten(space, obs):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).reshape(space.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return special.from_onehot(obs)
    elif isinstance(space, gym.spaces.Tuple):
        dims = [gym_space_flatten_dim(c) for c in space.spaces]
        flat_xs = np.split(obs, np.cumsum(dims)[:-1])
        return tuple(
            gym_space_unflatten(c, xi) for c, xi in zip(space.spaces, flat_xs))
    else:
        raise NotImplementedError


class NormalizedGymEnv(gym.Wrapper, Serializable):
    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            flatten_obs=True,
            obs_alpha=0.001,
            reward_alpha=0.001,
    ):
        Serializable.quick_init(self, locals())
        super(NormalizedGymEnv, self).__init__(env)
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._flatten_obs = flatten_obs

        self._obs_alpha = obs_alpha
        flat_obs_dim = gym_space_flatten_dim(env.observation_space)
        self._obs_mean = np.zeros(flat_obs_dim)
        self._obs_var = np.ones(flat_obs_dim)

        self._reward_alpha = reward_alpha
        self._reward_mean = 0.
        self._reward_var = 1.

    def _update_obs_estimate(self, obs):
        flat_obs = gym_space_flatten(self.env.observation_space, obs)
        self._obs_mean = (
            1 - self._obs_alpha) * self._obs_mean + self._obs_alpha * flat_obs
        self._obs_var = (
            1 - self._obs_alpha) * self._obs_var + self._obs_alpha * np.square(
                flat_obs - self._obs_mean)

    def _update_reward_estimate(self, reward):
        self._reward_mean = (1 - self._reward_alpha) * \
            self._reward_mean + self._reward_alpha * reward
        self._reward_var = (
            1 - self._reward_alpha
        ) * self._reward_var + self._reward_alpha * np.square(
            reward - self._reward_mean)

    def _apply_normalize_obs(self, obs):
        self._update_obs_estimate(obs)
        normalized_obs = (gym_space_flatten(self.env.observation_space, obs) -
                          self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)
        if not self._flatten_obs:
            normalized_obs = gym_space_unflatten(self.env.observation_space,
                                                 normalized_obs)
        return normalized_obs

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    @overrides
    def reset(self, **kwargs):
        ret = self.env.reset(**kwargs)
        if self._normalize_obs:
            return self._apply_normalize_obs(ret)
        else:
            return ret

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        d["_obs_mean"] = self._obs_mean
        d["_obs_var"] = self._obs_var
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)
        self._obs_mean = d["_obs_mean"]
        self._obs_var = d["_obs_var"]

    @overrides
    def step(self, action):
        if isinstance(self.action_space, gym.spaces.Box):
            # rescale the action
            lb, ub = self.action_space.low, self.action_space.high
            if lb != -np.inf or ub != -np.inf:
                scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
                scaled_action = np.clip(scaled_action, lb, ub)
            else:
                scaled_action = action
        else:
            scaled_action = action

        next_obs, reward, done, info = self.env.step(scaled_action)

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        return next_obs, reward * self._scale_reward, done, info

    def __str__(self):
        return "Normalized: %s" % self.env


normalize = NormalizedGymEnv
