"""Multitask environment."""

import gym
import numpy as np

from garage.core import Serializable


class MultitaskEnv(gym.Wrapper, Serializable):
    """
    Multitask Environment wrapper.

    This class take an task-based environment and wrap it.
    It provides functions like resetting/sampling tasks
    that is used for multitask sampling.
    """

    def __init__(self, wrapped_env, tasks, task_sample_method="round_robin"):
        """
        Initialize a MultitaskEnv.

        Args
            wrapped_env: the environment that is wrapped. Note that
                         this environment need a task space.
            tasks: a list of tasks to be sampled from.
            task_sample_method: how to sample/reset a task.
                          Options are: "round_robin" and "random".
        """
        assert wrapped_env.task_space

        super().__init__(wrapped_env)
        self.tasks = tasks
        self.n_tasks = len(tasks)
        self.task_sample_method = task_sample_method
        self.running_task_id = 0
        Serializable.quick_init(self, locals())

    def reset(self, reset_task=True):
        """
        Reset the environment.

        Args
            reset_tasks: whether reset the task of the wrapped environment.
        Return
            the observation from the wrapped environment after reset.
        """
        if reset_task:
            self.reset_task()
        return self.env.reset()

    def reset_task(self, task_id=None):
        """
        Sample a task and reset it with the wrapped environment.

        Args
            task_id: an id of the task to be set. If None, a tasks
                    will be sample with method of self.task_sample_method.
        """
        if task_id is not None:
            self.running_task_id = task_id
        elif self.task_sample_method == "round_robin":
            self.running_task_id = (self.running_task_id + 1) % self.n_tasks
        elif self.task_sample_method == "random":
            self.running_task_id = np.random.randint(low=0, high=self.n_tasks)
        else:
            raise NotImplementedError
        self.env.reset_task(self.tasks[self.running_task_id])
