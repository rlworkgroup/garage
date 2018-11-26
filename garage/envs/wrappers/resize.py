"""Resize wrapper for gym.Env."""
import gym
from gym.spaces import Box
import numpy as np
from scipy.misc import imresize


class Resize(gym.Wrapper):
    """gym.Env wrapper for resizing frame to (width, height)."""

    def __init__(self, env, width, height):
        """
        Resize wrapper.

        Args:
            env: gym.Env to wrap.
            width: resized frame width.
            height: resized frame height.
        """
        super().__init__(env)
        self.observation_space = Box(
            0.0, 1.0, shape=[width, height], dtype=np.float32)
        self.width = width
        self.height = height

    def _observation(self, obs):
        obs = imresize(obs, (self.width, self.height))
        return obs

    def reset(self):
        """gym.Env reset function."""
        return self._observation(self.env.reset())

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info
