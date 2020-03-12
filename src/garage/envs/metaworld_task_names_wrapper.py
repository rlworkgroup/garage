"""A wrapper for Metaworld environment to expose names of all tasks."""
import gym


class MetaworldTaskNamesWrapper(gym.Wrapper):
    """A wrapper for Metaworld environment to expose names of all tasks.

    Args:
        env (garage.envs.GarageEnv): An environment instance.

    """

    def __init__(self, env):
        super().__init__(env)
        self._hidden_env = env
        while True:
            if hasattr(self._hidden_env, 'env'):
                self._hidden_env = self._hidden_env.env
            else:
                break
        # By here, self._hidden_env should saves the underlying metaworld env

    @property
    def task_names(self):
        """list[str]: Name of all available tasks."""
        # pylint: disable=protected-access
        return self._hidden_env._task_names

    def step(self, action):
        """gym.Env step for the active task env.

        Args:
            action (np.ndarray): Action performed by the agent in the
                environment.

        Returns:
            np.ndarray: Agent's observation of the current environment.
            float: Amount of reward yielded by previous action.
            bool: True iff the episode has ended.
            dict[str, np.ndarray]: Contains auxiliary diagnostic
                information about this time-step.

        """
        obs, reward, done, info = self.env.step(action)
        info['task_name'] = self.task_name
        return obs, reward, done, info

    def set_task(self, task):
        """Set active task.

        Args:
            task (dict or int): A task.

        """
        # pylint: disable=protected-access, attribute-defined-outside-init
        self.env.set_task(task)
        task_id = self._hidden_env._active_task
        self.task_name = self.task_names[task_id]
