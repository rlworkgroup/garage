"""A wrapper env that handles multiple tasks from different envs.

Useful while training multi-task reinforcement learning algorithms.
It provides observations augmented with one-hot representation of tasks.
"""

import random

import akro
import gym
import numpy as np


def round_robin_strategy(num_tasks, last_task=None):
    """A function for sampling tasks in round robin fashion.

    Args:
        num_tasks (int): Total number of tasks.
        last_task (int): Previously sampled task.

    Returns:
        int: task id.

    """
    if last_task is None:
        return 0

    return (last_task + 1) % num_tasks


def uniform_random_strategy(num_tasks, _):
    """A function for sampling tasks uniformly at random.

    Args:
        num_tasks (int): Total number of tasks.
        _ (object): Ignored by this sampling strategy.

    Returns:
        int: task id.

    """
    return random.randint(0, num_tasks - 1)


class MultiEnvWrapper(gym.Wrapper):
    """A wrapper class to handle multiple gym environments.

    Args:
        envs (list(gym.Env)):
            A list of objects implementing gym.Env.
        sample_strategy (function(int, int)):
            Sample strategy to be used when sampling a new task.

    """

    def __init__(self, envs, sample_strategy=uniform_random_strategy):

        self._sample_strategy = sample_strategy
        self._num_tasks = len(envs)
        self._active_task_index = None
        self._observation_space = None

        super().__init__(envs[0])

        self._task_envs = []
        for env in envs:
            if (env.observation_space.shape !=
                    self.env.observation_space.shape):
                raise ValueError(
                    'Observation space of all envs should be same.')
            if env.action_space.shape != self.env.action_space.shape:
                raise ValueError('Action space of all envs should be same.')
            self._task_envs.append(env)

        self.env.spec.observation_space = self.observation_space

    @property
    def num_tasks(self):
        """Total number of tasks.

        Returns:
            int: number of tasks.

        """
        return len(self._task_envs)

    @property
    def task_space(self):
        """Task Space.

        Returns:
            akro.Box: Task space.

        """
        one_hot_ub = np.ones(self.num_tasks)
        one_hot_lb = np.zeros(self.num_tasks)
        return akro.Box(one_hot_lb, one_hot_ub)

    @property
    def active_task_index(self):
        """Index of active task env.

        Returns:
            int: Index of active task.

        """
        return self._active_task_index

    @property
    def observation_space(self):
        """Observation space.

        Returns:
            akro.Box: Observation space.

        """
        task_lb, task_ub = self.task_space.bounds
        env_lb, env_ub = self._observation_space.bounds
        return akro.Box(np.concatenate([task_lb, env_lb]),
                        np.concatenate([task_ub, env_ub]))

    @observation_space.setter
    def observation_space(self, observation_space):
        """Observation space setter.

        Args:
            observation_space (akro.Box): Observation space.

        """
        self._observation_space = observation_space

    @property
    def active_task_one_hot(self):
        """One-hot representation of active task.

        Returns:
            numpy.ndarray: one-hot representation of active task

        """
        one_hot = np.zeros(self.task_space.shape)
        index = self.active_task_index or 0
        one_hot[index] = self.task_space.high[index]
        return one_hot

    def reset(self, **kwargs):
        """Sample new task and call reset on new task env.

        Args:
            kwargs (dict): Keyword arguments to be passed to gym.Env.reset

        Returns:
            numpy.ndarray: active task one-hot representation + observation

        """
        self._active_task_index = self._sample_strategy(
            self._num_tasks, self._active_task_index)
        self.env = self._task_envs[self._active_task_index]
        obs = self.env.reset(**kwargs)
        oh_obs = self._obs_with_one_hot(obs)
        return oh_obs

    def step(self, action):
        """gym.Env step for the active task env.

        Args:
            action (object): object to be passed in gym.Env.reset(action)

        Returns:
            object: agent's observation of the current environment
            float: amount of reward returned after previous action
            bool: whether the episode has ended
            dict: contains auxiliary diagnostic information

        """
        obs, reward, done, info = self.env.step(action)
        oh_obs = self._obs_with_one_hot(obs)
        info['task_id'] = self._active_task_index
        return oh_obs, reward, done, info

    def close(self):
        """Close all task envs."""
        for env in self._task_envs:
            env.close()

    def _obs_with_one_hot(self, obs):
        """Concatenate active task one-hot representation with observation.

        Args:
            obs (numpy.ndarray): observation

        Returns:
            numpy.ndarray: active task one-hot + observation

        """
        oh_obs = np.concatenate([self.active_task_one_hot, obs])
        return oh_obs
