import copy

import gym
import numpy as np

from garage.experiment.core.misc import get_env_spec
from garage.experiment.loggers import Summary
from garage.experiment.samplers.sampler import Sampler


class BatchSampler(Sampler):
    """Multiple environments batch sampler."""

    def __init__(self, env: gym.Env, n_env=1, max_path_length=100):
        self._env = env
        self.n_env = n_env
        self.max_path_length = max_path_length

        self.envs = [copy.deepcopy(env) for _ in range(n_env)]
        self.path_idx = [i for i in range(self.n_env)]
        self.paths = []

        self.env_spec = get_env_spec(self._env)
        self.obs_dim = self.env_spec.observation_space.flat_dim
        self.action_dim = self.env_spec.action_space.flat_dim

        self._sample_count = 0
        self._path_count = 0
        self._last_obs = np.zeros((n_env, self.obs_dim))

    def _empty_path(self):
        # (observation, action, reward, info)
        return (np.zeros((0, self.obs_dim)), np.zeros((0, self.action_dim)),
                np.array([]), [])

    def reset(self):
        self.path_idx = [i for i in range(self.n_env)]
        self.paths = []
        self._path_count = self._sample_count = 0

        ret_obs = []

        for i in range(self.n_env):
            obs = self.envs[i].reset()
            ret_obs.append(obs)
            self._last_obs[i] = obs
            self.paths.append(self._empty_path())

        return np.array(ret_obs)

    def step(self, actions):
        ret_obs = []

        for i in range(self.n_env):
            idx = self.path_idx[i]
            env = self.envs[i]
            obs_all, action_all, rew_all, info_all = self.paths[idx]
            obs, rew, done, info = env.step(actions[i])
            ret_obs.append(obs)

            obs_all = np.vstack((obs_all, self._last_obs[i]))
            action_all = np.vstack((action_all, actions[i]))
            rew_all = np.append(rew_all, rew)
            info_all.append(info)
            self.paths[idx] = (obs_all, action_all, rew_all, info_all)

            if done:
                idx = idx + 1
                self.path_idx[i] = idx
                self.paths.append(self._empty_path())
                self._last_obs[i] = env.reset()
                self._path_count = self._path_count + 1
            else:
                self._last_obs[i] = obs
                self._sample_count = self._sample_count + 1

        return np.array(ret_obs)

    def get_samples(self):
        return [{
            'observations': obs.copy(),
            'actions': a.copy(),
            'rewards': rew.copy(),
            'infos': info.copy()
        } for (obs, a, rew, info) in self.paths]

    @property
    def sample_count(self):
        return self._sample_count

    @property
    def path_count(self):
        return self._path_count

    def get_summary(self):
        return Summary({
            'AverageReward': self.rewards.mean(),
            'StdReward': np.std(self.rewards),
            'MaxReward': np.max(self.rewards),
            'MinReward': np.min(self.rewards),
            'NumSteps': self.sample_count,
            'NumPaths': self.path_count,
        })

    @property
    def rewards(self):
        return np.concatenate([rews for _, _, rews, _ in self.paths])
