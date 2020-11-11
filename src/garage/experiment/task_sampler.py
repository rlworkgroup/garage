"""Efficient and general interfaces for sampling tasks for Meta-RL."""
# yapf: disable
import abc
import copy
import math

import numpy as np

from garage.envs import GymEnv, TaskNameWrapper, TaskOnehotWrapper
from garage.sampler.env_update import (ExistingEnvUpdate, NewEnvUpdate,
                                       SetTaskUpdate)

# yapf: enable


def _sample_indices(n_to_sample, n_available_tasks, with_replacement):
    """Select indices of tasks to sample.

    Args:
        n_to_sample (int): Number of environments to sample. May be greater
            than n_available_tasks.
        n_available_tasks (int): Number of available tasks. Task indices will
            be selected in the range [0, n_available_tasks).
        with_replacement (bool): Whether tasks can repeat when sampled.
            Note that if more tasks are sampled than exist, then tasks may
            repeat, but only after every environment has been included at
            least once in this batch. Ignored for continuous task spaces.

    Returns:
        np.ndarray[int]: Array of task indices.

    """
    if with_replacement:
        return np.random.randint(n_available_tasks, size=n_to_sample)
    else:
        blocks = []
        for _ in range(math.ceil(n_to_sample / n_available_tasks)):
            s = np.arange(n_available_tasks)
            np.random.shuffle(s)
            blocks.append(s)
        return np.concatenate(blocks)[:n_to_sample]


class TaskSampler(abc.ABC):
    """Class for sampling batches of tasks, represented as `~EnvUpdate`s.

    Attributes:
        n_tasks (int or None): Number of tasks, if known and finite.

    """

    @abc.abstractmethod
    def sample(self, n_tasks, with_replacement=False):
        """Sample a list of environment updates.

        Args:
            n_tasks (int): Number of updates to sample.
            with_replacement (bool): Whether tasks can repeat when sampled.
                Note that if more tasks are sampled than exist, then tasks may
                repeat, but only after every environment has been included at
                least once in this batch. Ignored for continuous task spaces.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """

    @property
    def n_tasks(self):
        """int or None: The number of tasks if known and finite."""
        return None


class ConstructEnvsSampler(TaskSampler):
    """TaskSampler where each task has its own constructor.

    Generally, this is used when the different tasks are completely different
    environments.

    Args:
        env_constructors (list[Callable[Environment]]): Callables that produce
            environments (for example, environment types).

    """

    def __init__(self, env_constructors):
        self._env_constructors = env_constructors

    @property
    def n_tasks(self):
        """int: the number of tasks."""
        return len(self._env_constructors)

    def sample(self, n_tasks, with_replacement=False):
        """Sample a list of environment updates.

        Args:
            n_tasks (int): Number of updates to sample.
            with_replacement (bool): Whether tasks can repeat when sampled.
                Note that if more tasks are sampled than exist, then tasks may
                repeat, but only after every environment has been included at
                least once in this batch. Ignored for continuous task spaces.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """
        return [
            NewEnvUpdate(self._env_constructors[i]) for i in _sample_indices(
                n_tasks, len(self._env_constructors), with_replacement)
        ]


class SetTaskSampler(TaskSampler):
    """TaskSampler where the environment can sample "task objects".

    This is used for environments that implement `sample_tasks` and `set_task`.
    For example, :py:class:`~HalfCheetahVelEnv`, as implemented in Garage.

    Args:
        env_constructor (type): Type of the environment.
        env (garage.Environment or None): Instance of env_constructor to sample
            from (will be constructed if not provided).
        wrapper (Callable[garage.Environment, garage.Environment] or None):
            Wrapper function to apply to environment.


    """

    def __init__(self, env_constructor, *, env=None, wrapper=None):
        self._env_constructor = env_constructor
        self._env = env or env_constructor()
        self._wrapper = wrapper

    @property
    def n_tasks(self):
        """int or None: The number of tasks if known and finite."""
        return getattr(self._env, 'num_tasks', None)

    def sample(self, n_tasks, with_replacement=False):
        """Sample a list of environment updates.

        Args:
            n_tasks (int): Number of updates to sample.
            with_replacement (bool): Whether tasks can repeat when sampled.
                Note that if more tasks are sampled than exist, then tasks may
                repeat, but only after every environment has been included at
                least once in this batch. Ignored for continuous task spaces.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """
        return [
            SetTaskUpdate(self._env_constructor, task, self._wrapper)
            for task in self._env.sample_tasks(n_tasks)
        ]


class EnvPoolSampler(TaskSampler):
    """TaskSampler that samples from a finite pool of environments.

    This can be used with any environments, but is generally best when using
    in-process samplers with environments that are expensive to construct.

    Args:
        envs (list[Environment]): List of environments to use as a pool.

    """

    def __init__(self, envs):
        self._envs = envs

    @property
    def n_tasks(self):
        """int: the number of tasks."""
        return len(self._envs)

    def sample(self, n_tasks, with_replacement=False):
        """Sample a list of environment updates.

        Args:
            n_tasks (int): Number of updates to sample.
            with_replacement (bool): Whether tasks can repeat when sampled.
                Since this cannot be easily implemented for an object pool,
                setting this to True results in ValueError.

        Raises:
            ValueError: If the number of requested tasks is larger than the
                pool, or with_replacement is set.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """
        if n_tasks > len(self._envs):
            raise ValueError('Cannot sample more environments than are '
                             'present in the pool. If more tasks are needed, '
                             'call grow_pool to copy random existing tasks.')
        if with_replacement:
            raise ValueError('EnvPoolSampler cannot meaningfully sample with '
                             'replacement.')
        envs = list(self._envs)
        np.random.shuffle(envs)
        return [ExistingEnvUpdate(env) for env in envs[:n_tasks]]

    def grow_pool(self, new_size):
        """Increase the size of the pool by copying random tasks in it.

        Note that this only copies the tasks already in the pool, and cannot
        create new original tasks in any way.

        Args:
            new_size (int): Size the pool should be after growning.

        """
        if new_size <= len(self._envs):
            return
        to_copy = _sample_indices(new_size - len(self._envs),
                                  len(self._envs),
                                  with_replacement=False)
        for idx in to_copy:
            self._envs.append(copy.deepcopy(self._envs[idx]))


MW_TASKS_PER_ENV = 50


class MetaWorldTaskSampler(TaskSampler):
    """TaskSampler that distributes a Meta-World benchmark across workers.

    Args:
        benchmark (metaworld.Benchmark): Benchmark to sample tasks from.
        kind (str): Must be either 'test' or 'train'. Determines whether to
            sample training or test tasks from the Benchmark.
        wrapper (Callable[garage.Env, garage.Env] or None): Wrapper to apply to
            env instances.
        add_env_onehot (bool): If true, a one-hot representing the current
            environment name will be added to the environments. Should only be
            used with multi-task benchmarks.

    Raises:
        ValueError: If kind is not 'train' or 'test'. Also raisd if
            `add_env_onehot` is used on a metaworld meta learning (not
            multi-task) benchmark.

    """

    def __init__(self, benchmark, kind, wrapper=None, add_env_onehot=False):
        self._benchmark = benchmark
        self._kind = kind
        self._inner_wrapper = wrapper
        self._add_env_onehot = add_env_onehot
        if kind == 'train':
            self._classes = benchmark.train_classes
            self._tasks = benchmark.train_tasks
        elif kind == 'test':
            self._classes = benchmark.test_classes
            self._tasks = benchmark.test_tasks
        else:
            raise ValueError('kind must be either "train" or "test", '
                             f'not {kind!r}')
        self._task_indices = {}
        if add_env_onehot:
            if kind == 'test' or 'metaworld.ML' in repr(type(benchmark)):
                raise ValueError('add_env_onehot should only be used with '
                                 f'multi-task benchmarks, not {benchmark!r}')
            self._task_indices = {
                env_name: index
                for (index, env_name) in enumerate(self._classes.keys())
            }
        self._task_map = {
            env_name:
            [task for task in self._tasks if task.env_name == env_name]
            for env_name in self._classes.keys()
        }
        for tasks in self._task_map.values():
            assert len(tasks) == MW_TASKS_PER_ENV
        self._task_orders = {
            env_name: np.arange(50)
            for env_name in self._task_map.keys()
        }
        self._next_order_index = 0
        self._shuffle_tasks()

    def _shuffle_tasks(self):
        """Reshuffles the task orders."""
        for tasks in self._task_orders.values():
            np.random.shuffle(tasks)

    @property
    def n_tasks(self):
        """int: the number of tasks."""
        return len(self._tasks)

    def sample(self, n_tasks, with_replacement=False):
        """Sample a list of environment updates.

        Note that this will always return environments in the same order, to
        make parallel sampling across workers efficient. If randomizing the
        environment order is required, shuffle the result of this method.

        Args:
            n_tasks (int): Number of updates to sample. Must be a multiple of
                the number of env classes in the benchmark (e.g. 1 for MT/ML1,
                10 for MT10, 50 for MT50). Tasks for each environment will be
                grouped to be adjacent to each other.
            with_replacement (bool): Whether tasks can repeat when sampled.
                Since this cannot be easily implemented for an object pool,
                setting this to True results in ValueError.

        Raises:
            ValueError: If the number of requested tasks is not equal to the
                number of classes or the number of total tasks.

        Returns:
            list[EnvUpdate]: Batch of sampled environment updates, which, when
                invoked on environments, will configure them with new tasks.
                See :py:class:`~EnvUpdate` for more information.

        """
        if n_tasks % len(self._classes) != 0:
            raise ValueError('For this benchmark, n_tasks must be a multiple '
                             f'of {len(self._classes)}')
        tasks_per_class = n_tasks // len(self._classes)
        updates = []

        # Avoid pickling the entire task sampler into every EnvUpdate
        inner_wrapper = self._inner_wrapper
        add_env_onehot = self._add_env_onehot
        task_indices = self._task_indices

        def wrap(env, task):
            """Wrap an environment in a metaworld benchmark.

            Args:
                env (gym.Env): A metaworld / gym environment.
                task (metaworld.Task): A metaworld task.

            Returns:
                garage.Env: The wrapped environment.

            """
            env = GymEnv(env, max_episode_length=env.max_path_length)
            env = TaskNameWrapper(env, task_name=task.env_name)
            if add_env_onehot:
                env = TaskOnehotWrapper(env,
                                        task_index=task_indices[task.env_name],
                                        n_total_tasks=len(task_indices))
            if inner_wrapper is not None:
                env = inner_wrapper(env, task)
            return env

        for env_name, env in self._classes.items():
            order_index = self._next_order_index
            for _ in range(tasks_per_class):
                task_index = self._task_orders[env_name][order_index]
                task = self._task_map[env_name][task_index]
                updates.append(SetTaskUpdate(env, task, wrap))
                if with_replacement:
                    order_index = np.random.randint(0, MW_TASKS_PER_ENV)
                else:
                    order_index += 1
                    order_index %= MW_TASKS_PER_ENV
        self._next_order_index += tasks_per_class
        if self._next_order_index >= MW_TASKS_PER_ENV:
            self._next_order_index %= MW_TASKS_PER_ENV
            self._shuffle_tasks()
        return updates
