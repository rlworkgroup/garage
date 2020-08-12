"""A wrapper env that handles multiple tasks from different envs.

Useful while training multi-task reinforcement learning algorithms.
It provides observations augmented with one-hot representation of tasks.
"""

import random

import akro
import numpy as np

from garage import EnvSpec, EnvStep, Wrapper


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


class MultiEnvWrapper(Wrapper):
    """A wrapper class to handle multiple environments.

    This wrapper adds an integer 'task_id' to env_info every timestep.

    Args:
        envs (list(Environment)):
            A list of objects implementing Environment.
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
        self._mode = mode

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
                    self._env.observation_space.shape):
                raise ValueError(
                    'Observation space of all envs should be same.')
            if env.action_space.shape != self._env.action_space.shape:
                raise ValueError('Action space of all envs should be same.')
            self._task_envs.append(env)

    @property
    def observation_space(self):
        """Observation space.

        Returns:
            akro.Box: Observation space.

        """
        if self._mode == 'vanilla':
            return self._env.observation_space
        elif self._mode == 'add-onehot':
            task_lb, task_ub = self.task_space.bounds
            env_lb, env_ub = self._env.observation_space.bounds
            return akro.Box(np.concatenate([env_lb, task_lb]),
                            np.concatenate([env_ub, task_ub]))
        else:  # self._mode == 'del-onehot'
            env_lb, env_ub = self._env.observation_space.bounds
            num_tasks = self._num_tasks
            return akro.Box(env_lb[:-num_tasks], env_ub[:-num_tasks])

    @property
    def spec(self):
        """Describes the action and observation spaces of the wrapped envs.

        Returns:
            EnvSpec: the action and observation spaces of the
                wrapped environments.

        """
        return EnvSpec(action_space=self.action_space,
                       observation_space=self.observation_space,
                       max_episode_length=self._env.spec.max_episode_length)

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
        if hasattr(self._env, 'active_task_index'):
            return self._env.active_task_index
        else:
            return self._active_task_index

    def reset(self):
        """Sample new task and call reset on new task env.

        Returns:
            numpy.ndarray: The first observation conforming to
                `observation_space`.
            dict: The episode-level information.
                Note that this is not part of `env_info` provided in `step()`.
                It contains information of he entire episode， which could be
                needed to determine the first action (e.g. in the case of
                goal-conditioned or MTRL.)

        """
        self._active_task_index = self._sample_strategy(
            self._num_tasks, self._active_task_index)
        self._env = self._task_envs[self._active_task_index]
        obs, episode_info = self._env.reset()

        if self._mode == 'vanilla':
            pass
        elif self._mode == 'add-onehot':
            obs = np.concatenate([obs, self._active_task_one_hot()])
        else:  # self._mode == 'del-onehot'
            obs = obs[:-self._num_tasks]

        return obs, episode_info

    def step(self, action):
        """Step the active task env.

        Args:
            action (object): object to be passed in Environment.reset(action)

        Returns:
            EnvStep: The environment step resulting from the action.

        """
        es = self._env.step(action)

        if self._mode == 'add-onehot':
            obs = np.concatenate([es.observation, self._active_task_one_hot()])
        elif self._mode == 'del-onehot':
            obs = es.observation[:-self._num_tasks]
        else:  # self._mode == 'vanilla'
            obs = es.observation

        env_info = es.env_info
        if 'task_id' not in es.env_info:
            env_info['task_id'] = self._active_task_index
        if self._env_names is not None:
            env_info['task_name'] = self._env_names[self._active_task_index]

        return EnvStep(env_spec=self.spec,
                       action=action,
                       reward=es.reward,
                       observation=obs,
                       env_info=env_info,
                       step_type=es.step_type)

    def close(self):
        """Close all task envs."""
        for env in self._task_envs:
            env.close()

    def _active_task_one_hot(self):
        """One-hot representation of active task.

        Returns:
            numpy.ndarray: one-hot representation of active task

        """
        one_hot = np.zeros(self.task_space.shape)
        index = self.active_task_index or 0
        one_hot[index] = self.task_space.high[index]
        return one_hot
