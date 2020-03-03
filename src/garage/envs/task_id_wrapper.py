"""Wrapper for Metaworld environment to provide properly-named tasks."""

import gym


class TaskIdWrapper(gym.Wrapper):
    """Wrapper for Metaworld environment to provide properly-named tasks."""

    @property
    def _hidden_env(self):
        """gym.Env: The underlying Metaworld environment."""
        env = self.env
        while hasattr(env, 'env'):
            env = env.env
        return env

    @property
    def task_names(self):
        """list[str]: List of all available names."""
        # pylint: disable=protected-access
        return self._hidden_env._task_names

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
        info['task_id'] = self.task_id
        info['task_name'] = self.task_name
        return obs, reward, done, info

    def set_task(self, task):
        """Set active task.

        Args:
            task (dict or int): A task.

        """
        # pylint: disable=protected-access, attribute-defined-outside-init
        self.env.set_task(task)
        self.task_id = self._hidden_env._active_task
        self.task_name = self._hidden_env._task_names[self.task_id]
