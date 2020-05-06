"""A wrapper env that handles multiple tasks from different envs.

Useful while training multi-task reinforcement learning algorithms.
It provides observations augmented with one-hot representation of tasks.
"""

import random

import akro
import gym
import numpy as np

from garage.envs.garage_env import GarageEnv


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
    """A wrapper class to handle multiple environments.

    This wrapper adds an integer 'task_id' to env_info every timestep.

    Args:
        envs (list(gym.Env)):
            A list of objects implementing gym.Env.
        sample_strategy (function(int, int)):
            Sample strategy to be used when sampling a new task.
        mode (str): A string from 'vanilla`, 'add-onehot' and 'del-onehot'.
            The type of observation to use.
            - 'vanilla' provides the observation as it is.
              Use case: metaworld environments with MT* algorithms,
                        gym environments with Task Embedding.
            - 'add-onehot' will append an one-hot task id to observation.
              Use case: gym environments with MT* algorithms.
            - 'del-onehot' assumes an one-hot task id is appended to
              observation, and it excludes that.
              Use case: metaworld environments with Task Embedding.
        env_names (list(str)): The names of the environments corresponding to
            envs. The index of an env_name must correspond to the index of the
            corresponding env in envs. An env_name in env_names must be unique.

    """

    def __init__(self,
                 envs,
                 sample_strategy=uniform_random_strategy,
                 mode='add-onehot',
                 env_names=None):
        assert mode in ['vanilla', 'add-onehot', 'del-onehot']

        self._sample_strategy = sample_strategy
        self._num_tasks = len(envs)
        self._active_task_index = None
        self._observation_space = None
        self._mode = mode
        for i, env in enumerate(envs):
            if not isinstance(env, GarageEnv):
                envs[i] = GarageEnv(env)
        super().__init__(envs[0])
        if env_names is not None:
            assert isinstance(env_names, list), 'env_names must be a list'
            msg = ('env_names are not unique or there is not an env_name',
                   'corresponding to each env in envs')
            assert len(set(env_names)) == len(envs), msg
        self._env_names = env_names
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
        self._spec = self.env.spec

    @property
    def spec(self):
        """Describes the action and observation spaces of the wrapped envs.

        Returns:
            garage.envs.EnvSpec: the action and observation spaces of the
                wrapped environments.

        """
        return self._spec

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
        if hasattr(self.env, 'active_task_index'):
            return self.env.active_task_index
        else:
            return self._active_task_index

        return self._active_task_index

    @property
    def observation_space(self):
        """Observation space.

        Returns:
            akro.Box: Observation space.

        """
        if self._mode == 'vanilla':
            return self.env.observation_space
        elif self._mode == 'add-onehot':
            task_lb, task_ub = self.task_space.bounds
            env_lb, env_ub = self._observation_space.bounds
            return akro.Box(np.concatenate([env_lb, task_lb]),
                            np.concatenate([env_ub, task_ub]))
        else:  # self._mode == 'del-onehot'
            env_lb, env_ub = self._observation_space.bounds
            num_tasks = self._num_tasks
            return akro.Box(env_lb[:-num_tasks], env_ub[:-num_tasks])

    @observation_space.setter
    def observation_space(self, observation_space):
        """Observation space setter.

        Args:
            observation_space (akro.Box): Observation space.

        """
        self._observation_space = observation_space

    def _active_task_one_hot(self):
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
        if self._mode == 'vanilla':
            return obs
        elif self._mode == 'add-onehot':
            return np.concatenate([obs, self._active_task_one_hot()])
        else:  # self._mode == 'del-onehot'
            return obs[:-self._num_tasks]

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
        if self._mode == 'add-onehot':
            obs = np.concatenate([obs, self._active_task_one_hot()])
        elif self._mode == 'del-onehot':
            obs = obs[:-self._num_tasks]
        if 'task_id' not in info:
            info['task_id'] = self._active_task_index
        if self._env_names is not None:
            info['task_name'] = self._env_names[self._active_task_index]
        return obs, reward, done, info

    def close(self):
        """Close all task envs."""
        for env in self._task_envs:
            env.close()
