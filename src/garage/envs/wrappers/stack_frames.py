"""Stack frames wrapper for gym.Env."""
from collections import deque

import gym
import gym.spaces
import numpy as np


class StackFrames(gym.Wrapper):
    """gym.Env wrapper to stack multiple frames.

    Useful for training feed-forward agents on dynamic games.
    Only works with gym.spaces.Box environment with 2D single channel frames.

    Args:
        env: gym.Env to wrap.
        n_frames: number of frames to stack.

    Raises:
        ValueError: If observation space shape is not 2 or
        environment is not gym.spaces.Box.

    """

    def __init__(self, env, n_frames):
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError('Stack frames only works with gym.spaces.Box '
                             'environment.')

        if len(env.observation_space.shape) != 2:
            raise ValueError(
                'Stack frames only works with 2D single channel images')

        super().__init__(env)

        self._n_frames = n_frames
        self._frames = deque(maxlen=n_frames)

        new_obs_space_shape = env.observation_space.shape + (n_frames, )
        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        self._observation_space = gym.spaces.Box(
            _low,
            _high,
            shape=new_obs_space_shape,
            dtype=env.observation_space.dtype)

    @property
    def observation_space(self):
        """gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _stack_frames(self):
        return np.stack(self._frames, axis=2)

    def reset(self):
        """gym.Env reset function."""
        observation = self.env.reset()
        self._frames.clear()
        for i in range(self._n_frames):
            self._frames.append(observation)

        return self._stack_frames()

    def step(self, action):
        """gym.Env step function."""
        new_observation, reward, done, info = self.env.step(action)
        self._frames.append(new_observation)

        return self._stack_frames(), reward, done, info
