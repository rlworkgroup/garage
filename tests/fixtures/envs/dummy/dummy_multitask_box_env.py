"""Dummy gym.spaces.Box environment for testing purpose."""
from random import choices

from tests.fixtures.envs.dummy import DummyBoxEnv


class DummyMultiTaskBoxEnv(DummyBoxEnv):
    """A dummy gym.spaces.Box multitask environment.

    Args:
        random (bool): If observations are randomly generated or not.
        obs_dim (iterable): Observation space dimension.
        action_dim (iterable): Action space dimension.

    """

    def __init__(self, random=True, obs_dim=(4, ), action_dim=(2, )):
        super().__init__(random, obs_dim, action_dim)
        self.task = 'dummy1'

    def sample_tasks(self, n):
        """Sample a list of `num_tasks` tasks.

        Args:
            n (int): Number of tasks to sample.

        Returns:
            list[str]: A list of tasks.

        """
        return choices(self.all_task_names, k=n)

    @property
    def all_task_names(self):
        """list[str]: Return a list of dummy task names."""
        return ['dummy1', 'dummy2', 'dummy3']

    def set_task(self, task):
        """Reset with a task.

        Args:
            task (str): A task.

        """
        self.task = task

    def step(self, action):
        """Step the environment.

        Args:
            action (int): Action input.

        Returns:
            np.ndarray: Observation.
            float: Reward.
            bool: If the environment is terminated.
            dict: Environment information.

        """
        return (self.observation_space.sample(), 0, False,
                dict(dummy='dummy', task_name=self.task))
