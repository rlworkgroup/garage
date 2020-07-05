"""Grayscale wrapper for gym.Env."""
import warnings

import gym
import gym.spaces
import numpy as np
from skimage import color, img_as_ubyte


class Grayscale(gym.Wrapper):
    """Grayscale wrapper for gym.Env, converting frames to grayscale.

    Only works with gym.spaces.Box environment with 2D RGB frames.
    The last dimension (RGB) of environment observation space will be removed.

    Example:
        env = gym.make('Env')
        # env.observation_space = (100, 100, 3)

        env_wrapped = Grayscale(gym.make('Env'))
        # env.observation_space = (100, 100)

    Args:
        env (gym.Env): Environment to wrap.

    Raises:
        ValueError: If observation space shape is not 3
            or environment is not gym.spaces.Box.

    """

    def __init__(self, env):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError(
                'Grayscale only works with gym.spaces.Box environment.')

        if len(env.observation_space.shape) != 3:
            raise ValueError('Grayscale only works with 2D RGB images')

        super().__init__(env)

        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        assert _low == 0
        assert _high == 255
        self._observation_space = gym.spaces.Box(
            _low,
            _high,
            shape=env.observation_space.shape[:-1],
            dtype=np.uint8)

    @property
    def observation_space(self):
        """gym.Env: Observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def reset(self, **kwargs):
        """gym.Env reset function.

        Args:
            **kwargs: Unused.

        Returns:
            np.ndarray: Observation conforming to observation_space
        """
        del kwargs
        return _color_to_grayscale(self.env.reset())

    def step(self, action):
        """See gym.Env.

        Args:
            action (np.ndarray): Action conforming to action_space

        Returns:
            np.ndarray: Observation conforming to observation_space
            float: Reward for this step
            bool: Termination signal
            dict: Extra information from the environment.

        """
        obs, reward, done, info = self.env.step(action)
        return _color_to_grayscale(obs), reward, done, info


def _color_to_grayscale(obs):
    """Convert a 3-channel color observation image to grayscale and uint8.

    Args:
       obs (np.ndarray): Observation array, conforming to observation_space

    Returns:
       np.ndarray: 1-channel grayscale version of obs, represented as uint8

    """
    with warnings.catch_warnings():
        # Suppressing warning for possible precision loss when converting
        # from float64 to uint8
        warnings.simplefilter('ignore')
        return img_as_ubyte(color.rgb2gray((obs)))
