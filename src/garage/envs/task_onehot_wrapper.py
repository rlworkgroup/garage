"""Wrapper for appending one-hot task encodings to individual task envs.

See `~TaskOnehotWrapper.wrap_env_list` for the main way of using this module.

"""
import akro
import gym
import numpy as np

from garage.envs.env_spec import EnvSpec


class TaskOnehotWrapper(gym.Wrapper):
    """Append a one-hot task representation to an environment.

    See TaskOnehotWrapper.wrap_env_list for the recommended way of creating
    this class.

    Args:
        env (gym.Env): The environment to wrap.
        task_index (int): The index of this task among the tasks.
        n_total_tasks (int): The number of total tasks.

    """

    def __init__(self, env, task_index, n_total_tasks):
        assert 0 <= task_index < n_total_tasks
        super().__init__(env)
        self._task_index = task_index
        self._n_total_tasks = n_total_tasks
        env_lb = self.env.observation_space.low
        env_ub = self.env.observation_space.high
        one_hot_ub = np.ones(self._n_total_tasks)
        one_hot_lb = np.zeros(self._n_total_tasks)
        self.observation_space = akro.Box(np.concatenate([env_lb, one_hot_lb]),
                                          np.concatenate([env_ub, one_hot_ub]))
        self.__spec = EnvSpec(action_space=self.action_space,
                              observation_space=self.observation_space)

    @property
    def spec(self):
        """Return the environment specification.

        Returns:
            garage.envs.env_spec.EnvSpec: The envionrment specification.

        """
        return self.__spec

    def reset(self, **kwargs):
        """Sample new task and call reset on new task env.

        Args:
            kwargs (dict): Keyword arguments to be passed to env.reset

        Returns:
            numpy.ndarray: active task one-hot representation + observation

        """
        return self._obs_with_one_hot(self.env.reset(**kwargs))

    def step(self, action):
        """gym.Env step for the active task env.

        Args:
            action (np.ndarray): Action performed by the agent in the
                environment.

        Returns:
            tuple:
                np.ndarray: Agent's observation of the current environment.
                float: Amount of reward yielded by previous action.
                bool: True iff the episode has ended.
                dict[str, np.ndarray]: Contains auxiliary diagnostic
                    information about this time-step.

        """
        obs, reward, done, info = self.env.step(action)
        oh_obs = self._obs_with_one_hot(obs)
        info['task_id'] = self._task_index
        return oh_obs, reward, done, info

    def _obs_with_one_hot(self, obs):
        """Concatenate observation and task one-hot.

        Args:
            obs (numpy.ndarray): observation

        Returns:
            numpy.ndarray: observation + task one-hot.

        """
        one_hot = np.zeros(self._n_total_tasks)
        one_hot[self._task_index] = 1.0
        return np.concatenate([obs, one_hot])

    @classmethod
    def wrap_env_list(cls, envs):
        """Wrap a list of environments, giving each environment a one-hot.

        This is the primary way of constructing instances of this class.
        It's mostly useful when training multi-task algorithms using a
        multi-task aware sampler.

        For example:
        '''
        .. code-block:: python

            envs = get_mt10_envs()
            wrapped = TaskOnehotWrapper.wrap_env_list(envs)
            sampler = runner.make_sampler(LocalSampler, env=wrapped)
        '''

        Args:
            envs (list[gym.Env]): List of environments to wrap. Note that the
                order these environments are passed in determines the value of
                their one-hot encoding. It is essential that this list is
                always in the same order, or the resulting encodings will be
                inconsistent.

        Returns:
            list[TaskOnehotWrapper]: The wrapped environments.

        """
        n_total_tasks = len(envs)
        wrapped = []
        for i, env in enumerate(envs):
            wrapped.append(cls(env, task_index=i, n_total_tasks=n_total_tasks))
        return wrapped

    @classmethod
    def wrap_env_cons_list(cls, env_cons):
        """Wrap a list of environment constructors, giving each a one-hot.

        This function is useful if you want to avoid constructing any
        environments in the main experiment process, and are using a multi-task
        aware remote sampler (i.e. `~RaySampler`).

        For example:
        '''
        .. code-block:: python

            env_constructors = get_mt10_env_cons()
            wrapped = TaskOnehotWrapper.wrap_env_cons_list(env_constructors)
            env_updates = [NewEnvUpdate(wrapped_con)
                           for wrapped_con in wrapped]
            sampler = runner.make_sampler(RaySampler, env=env_updates)
        '''


        Args:
            env_cons (list[Callable[gym.Env]]): List of environment constructor
                to wrap. Note that the order these constructors are passed in
                determines the value of their one-hot encoding. It is essential
                that this list is always in the same order, or the resulting
                encodings will be inconsistent.

        Returns:
            list[Callable[TaskOnehotWrapper]]: The wrapped environments.

        """
        n_total_tasks = len(env_cons)
        wrapped = []
        for i, con in enumerate(env_cons):
            # Manually capture this value of i by introducing a new scope.
            wrapped.append(lambda i=i, con=con: cls(
                con(), task_index=i, n_total_tasks=n_total_tasks))
        return wrapped
