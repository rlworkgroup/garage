"""Test environment wrapper."""

import gym


class AutoStopEnv(gym.Wrapper):
    """Environment wrapper that stops episode at step max_episode_length."""

    def __init__(self, env=None, env_name='', max_episode_length=100):
        """Create an AutoStepEnv.

        Args:
            env (gym.Env): Environment to be wrapped.
            env_name (str): Name of the environment.
            max_episode_length (int): Maximum length of the episode.
        """
        if env_name:
            super().__init__(gym.make(env_name))
        else:
            super().__init__(env)

        self._episode_step = 0
        self._max_episode_length = max_episode_length

    def step(self, action):
        """Step the wrapped environment.

        Args:
            action (np.ndarray): the action.

        Returns:
            np.ndarray: Next observation
            float: Reward
            bool: Termination signal
            dict: Environment information
        """
        self._episode_step += 1
        next_obs, reward, done, info = self.env.step(action)
        if self._episode_step == self._max_episode_length:
            done = True
            self._episode_step = 0
        return next_obs, reward, done, info

    def reset(self, **kwargs):
        """Reset the wrapped environment.

        Args:
            **kwargs: Keyword arguments.

        Returns:
            np.ndarray: Initial observation.
        """
        return self.env.reset(**kwargs)
