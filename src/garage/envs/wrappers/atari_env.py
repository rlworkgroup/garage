"""Episodic life wrapper for gym.Env."""
import gym
import numpy as np


class AtariEnv(gym.Wrapper):
    """Atari environment wrapper for gym.Env.

    This wrapper convert the observations returned from baselines wrapped
    environment, which is a LazyFrames object into numpy arrays.

    Args:
        env (gym.Env): The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return np.asarray(obs), reward, done, info

    def reset(self, **kwargs):
        """gym.Env reset function."""
        return np.asarray(self.env.reset())
