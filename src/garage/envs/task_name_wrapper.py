"""Wrapper for adding an environment info to track task ID."""
from garage import Wrapper


class TaskNameWrapper(Wrapper):
    """Add task_name or task_id to env infos.

    Args:
        env (gym.Env): The environment to wrap.
        task_name (str or None): Task name to be added, if any.
        task_id (int or None): Task ID to be added, if any.

    """

    def __init__(self, env, *, task_name=None, task_id=None):
        super().__init__(env)
        self._task_name = task_name
        self._task_id = task_id

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
        es = super().step(action)
        if self._task_name is not None:
            es.env_info['task_name'] = self._task_name
        if self._task_id is not None:
            es.env_info['task_id'] = self._task_id
        return es
