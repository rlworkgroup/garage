"""Episodic life wrapper for gym.Env."""
import gym


class EpisodicLife(gym.Wrapper):
    """Episodic life wrapper for gym.Env.

    This wrapper makes episode end when a life is lost, but only reset
    when all lives are lost.

    Args:
        env: The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)
        self._lives = 0
        self._was_real_done = True

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        self._was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self._lives and lives > 0:
            done = True
        self._lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """
        gym.Env reset function.

        Reset only when lives are lost.
        """
        if self._was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step
            obs, _, _, _ = self.env.step(0)
        self._lives = self.env.unwrapped.ale.lives()
        return obs
