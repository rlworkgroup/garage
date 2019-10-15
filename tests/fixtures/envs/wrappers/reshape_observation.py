"""Reshaping Observation for gym.Env."""
import gym
import numpy as np


class ReshapeObservation(gym.Wrapper):
    """
    Reshaping Observation wrapper for gym.Env.

    This wrapper convert the observations into the given shape.

    Args:
        env (gym.Env): The environment to be wrapped.
        shape (list[int]): Target shape to be applied on the observations.
    """

    def __init__(self, env, shape):
        super().__init__(env)
        print(env.observation_space.shape)
        assert np.prod(shape) == np.prod(env.observation_space.shape)
        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        self._observation_space = gym.spaces.Box(
            _low, _high, shape=shape, dtype=env.observation_space.dtype)
        self._shape = shape

    @property
    def observation_space(self):
        """gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _observation(self, obs):
        return obs.reshape(self._shape)

    def reset(self):
        """gym.Env reset function."""
        return self._observation(self.env.reset())

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info
