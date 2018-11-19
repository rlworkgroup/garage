"""Grayscale wrapper for gym.Env."""
import gym
from gym.spaces import Box
import numpy as np

# from skimage import color


class GrayScale(gym.Wrapper):
    """Converting frames to grayscale."""

    def __init__(self, env):
        """
        Grayscale wrapper.

        Args:
            env: gym.Env to wrap.
        """
        super(GrayScale, self).__init__(env)
        width = env.observation_space.shape[0]
        height = env.observation_space.shape[1]
        self.observation_space = Box(
            0.0, 1.0, shape=[width, height], dtype=np.float32)

    def _observation(self, obs):
        # obs = color.rgb2gray(obs) / 255.0
        obs = np.dot(obs[:, :, :3], [0.299, 0.587, 0.114]) / 255.0
        return obs

    def reset(self):
        """gym.Env reset function."""
        return self._observation(self.env.reset())

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info
