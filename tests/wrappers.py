"""Test environment wrapper."""

import gym


class AutoStopEnv(gym.Wrapper):
    """A env wrapper that stops rollout at step max_episode_length."""

    def __init__(self, env=None, env_name='', max_episode_length=100):
        """Create an AutoStepEnv.

        Args:
            env (gym.Env): the environment to be wrapped.
            env_name (str): the name of the environment.
            max_episode_length (int): the max step length of the episode.
        """
        if env_name:
            super().__init__(gym.make(env_name))
        else:
            super().__init__(env)
        self._rollout_step = 0
        self._max_episode_length = max_episode_length

    def step(self, action):
        """Step the wrapped environment.

        Args:
            action (np.ndarray): the action.

        Returns:
            np.ndarray: the next observation
            float: the reward
            bool: the termination signal
            dict: any environment information
        """
        self._rollout_step += 1
        next_obs, reward, done, info = self.env.step(action)
        if self._rollout_step == self._max_episode_length:
            done = True
            self._rollout_step = 0
        return next_obs, reward, done, info

    def reset(self, **kwargs):
        """Reset the wrapped environment.

        Args:
            **kwargs: keyword arguments.

        Returns:
            np.ndarray: the initial observation.
        """
        return self.env.reset(**kwargs)
