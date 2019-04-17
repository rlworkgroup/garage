"""Clip reward for gym.Env."""
import gym
import numpy as np


class ClipReward(gym.Wrapper):
    """Clip the reward by its sign."""

    def step(self, ac):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(ac)
        return obs, np.sign(reward), done, info

    def reset(self):
        """gym.Env reset."""
        return self.env.reset()
