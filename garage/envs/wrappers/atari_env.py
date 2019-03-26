"""Episodic life wrapper for gym.Env."""
import gym
import numpy as np


class AtariEnv(gym.Wrapper):
    """
    Episodic life wrapper for gym.Env.

    This wrapper makes episode end when a life is lost, but only reset
    when all lives are lost.

    Args:
        env: The environment to be wrapped.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return np.asarray(obs), reward, done, info

    def reset(self, **kwargs):
        """
        gym.Env reset function.

        Reset only when lives are lost.
        """
        return np.asarray(self.env.reset())
