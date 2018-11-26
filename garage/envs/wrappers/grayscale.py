"""Grayscale wrapper for gym.Env."""
import gym
from gym.spaces import Box
import numpy as np
from skimage import color


class Grayscale(gym.Wrapper):
    """
    Grayscale wrapper for gym.Env, converting frames to grayscale.

    Only works with Box environment with 2D RGB frames.
    The last dimension (RGB) of environment observation space will be removed.

    Example:
        env = gym.make('Env')
        # env.observation_space = (100, 100, 3)

        env_wrapped = Grayscale(gym.make('Env'))
        # env.observation_space = (100, 100)

    Args:
        env: gym.Env to wrap.

    Raises:
        ValueError: If observation space shape is not 3 or environment is not Box.

    """

    def __init__(self, env):
        if not isinstance(env.observation_space, Box):
            raise ValueError("Grayscale only works with Box environment.")

        if len(env.observation_space.shape) != 3:
            raise ValueError("Grayscale only works with 2D RGB images")

        super().__init__(env)

        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        self._observation_space = Box(
            _low,
            _high,
            shape=env.observation_space.shape[:-1],
            dtype=np.float32)

    @property
    def observation_space(self):
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _observation(self, obs):
        obs = color.rgb2gray(np.asarray(obs, dtype=np.uint8))
        return obs

    def reset(self):
        """gym.Env reset function."""
        return self._observation(self.env.reset())

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info
