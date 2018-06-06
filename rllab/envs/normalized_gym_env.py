import gym
import numpy as np

from rllab.core.serializable import Serializable
from rllab.misc import special
from rllab.misc.overrides import overrides


def gym_space_flatten_dim(space):
    if isinstance(space, gym.spaces.Box):
        return np.prod(space.low.shape)
    elif isinstance(space, gym.spaces.Discrete):
        return space.n
    elif isinstance(space, gym.spaces.Tuple):
        return np.sum([gym_space_flatten_dim(x) for x in space.spaces])
    else:
        raise NotImplementedError


def flatten_gym_space(obs, space):
    if isinstance(space, gym.spaces.Box):
        return np.asarray(obs).flatten()
    elif isinstance(space, gym.spaces.Discrete):
        return special.to_onehot(obs, space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return np.concatenate(
            [flatten_gym_space(xi, c) for c, xi in zip(space.spaces, obs)])
    else:
        raise NotImplementedError


class NormalizedGymEnv(gym.Env, Serializable):
    def __init__(
            self,
            env,
            scale_reward=1.,
            normalize_obs=False,
            normalize_reward=False,
            obs_alpha=0.001,
            reward_alpha=0.001,
    ):
        Serializable.quick_init(self, locals())
        super(NormalizedGymEnv, self).__init__()
        self._env = env
        self._scale_reward = scale_reward
        self._normalize_obs = normalize_obs
        self._normalize_reward = normalize_reward
        self._obs_alpha = obs_alpha

        flat_obs_dim = gym_space_flatten_dim(env.observation_space)

        self._obs_mean = np.zeros(flat_obs_dim)
        self._obs_var = np.ones(flat_obs_dim)
        self._reward_alpha = reward_alpha

        self._reward_mean = 0.
        self._reward_var = 1.

        self.action_space = self._env.action_space
        self.observation_space = self._env.observation_space
        self.spec = self._env.spec

    def _update_obs_estimate(self, obs):
        flat_obs = flatten_gym_space(obs, self._env.observation_space)
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
        return (flatten_gym_space(obs, self._env.observation_space) -
                self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

    def _apply_normalize_reward(self, reward):
        self._update_reward_estimate(reward)
        return reward / (np.sqrt(self._reward_var) + 1e-8)

    @overrides
    def reset(self):
        ret = self._env.reset()
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
        if isinstance(self._env.action_space, gym.spaces.Box):
            # rescale the action
            lb, ub = self._env.action_space.bounds
            scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
            scaled_action = np.clip(scaled_action, lb, ub)
        else:
            scaled_action = action

        next_obs, reward, done, info = self._env.step(scaled_action)

        if self._normalize_obs:
            next_obs = self._apply_normalize_obs(next_obs)
        if self._normalize_reward:
            reward = self._apply_normalize_reward(reward)

        return next_obs, reward * self._scale_reward, done, info

    @overrides
    def render(self, mode='human'):
        self._env.render()

    @overrides
    def close(self):
        self._env.close()

    @overrides
    def seed(self, seed=None):
        self._env.seed(seed)

    def __str__(self):
        return "Normalized: %s" % self._env


normalized_gym = NormalizedGymEnv
