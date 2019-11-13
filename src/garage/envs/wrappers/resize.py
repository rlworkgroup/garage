"""Resize wrapper for gym.Env."""
import warnings

import gym
import gym.spaces
import numpy as np
from skimage import img_as_ubyte
from skimage.transform import resize


class Resize(gym.Wrapper):
    """gym.Env wrapper for resizing frame to (width, height).

    Only works with gym.spaces.Box environment with 2D single channel frames.

    Example:
        | env = gym.make('Env')
        | # env.observation_space = (100, 100)
        | env_wrapped = Resize(gym.make('Env'), width=64, height=64)
        | # env.observation_space = (64, 64)

    Args:
        env: gym.Env to wrap.
        width: resized frame width.
        height: resized frame height.

    Raises:
        ValueError: If observation space shape is not 2
            or environment is not gym.spaces.Box.

    """

    def __init__(self, env, width, height):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError('Resize only works with Box environment.')

        if len(env.observation_space.shape) != 2:
            raise ValueError('Resize only works with 2D single channel image.')

        super().__init__(env)

        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        self._dtype = env.observation_space.dtype
        self._observation_space = gym.spaces.Box(_low,
                                                 _high,
                                                 shape=[width, height],
                                                 dtype=self._dtype)

        self._width = width
        self._height = height

    @property
    def observation_space(self):
        """gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _observation(self, obs):
        with warnings.catch_warnings():
            """
            Suppressing warnings for
            1. possible precision loss when converting from float64 to uint8
            2. anti-aliasing will be enabled by default in skimage 0.15
            """
            warnings.simplefilter('ignore')
            obs = resize(obs, (self._width, self._height))  # now it's float
            if self._dtype == np.uint8:
                obs = img_as_ubyte(obs)
        return obs

    def reset(self):
        """gym.Env reset function."""
        return self._observation(self.env.reset())

    def step(self, action):
        """gym.Env step function."""
        obs, reward, done, info = self.env.step(action)
        return self._observation(obs), reward, done, info
