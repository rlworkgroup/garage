"""A wrapper for Metaworld MultiTask Benchmarks."""
import gym

from garage.envs import GarageEnv, round_robin_strategy


class MTMetaWorldWrapper(gym.Wrapper):
    """A Wrapper for Metaworld MultiTask benchmarks.

        The Environments in the benchmark class should be constructed using the
        `from_task` API of Mtaworld benchmark environments.

    Args:
        envs (list(gym.Env)):
            A list of objects implementing gym.Env.

    """

    def __init__(self, envs):

        self._sample_strategy = round_robin_strategy
        self._num_tasks = len(envs)
        self._active_task_index = None
        for i, env in enumerate(envs):
            if not isinstance(env, GarageEnv):
                envs[i] = GarageEnv(env)
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

    @property
    def num_tasks(self):
        """Total number of tasks.

        Returns:
            int: number of tasks.

        """
        return len(self._task_envs)

    @property
    def task_names(self):
        """The names of all tasks in this MTMetaWorldWrapper instance.

        Returns:
            list(str): names of all task in t his MTMetaWorldWrapper instance.

        """
        task_names = []
        for env in self._task_envs:
            task_names.append(env.all_task_names[0])
        return task_names

    @property
    def active_task_id(self):
        """Static task ID of active task env, defined by metaworld.

        Returns:
            int: Static task ID of active task.

        """
        return self.env.active_task

    @property
    def active_task_one_hot(self):
        """One-hot representation of active task.

        Returns:
            numpy.ndarray: one-hot representation of active task

        """
        return self.env.active_task_one_hot

    @property
    def active_task_name(self):
        """Name of the active task.

        Returns:
            str: active task name given by Metaworld benchmark.

        """
        return self.env.all_task_names[0]

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
        return obs

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
        info['task_name'] = self.active_task_name
        info['task_id'] = self.active_task_id
        return obs, reward, done, info

    def close(self):
        """Close all task envs."""
        for env in self._task_envs:
            env.close()
