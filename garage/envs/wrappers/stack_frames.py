"""Stack frames wrapper for gym.Env."""
from collections import deque

import gym
from gym.spaces import Box
import numpy as np


class StackFrames(gym.Wrapper):
    """
    Stack frames wrapper.

    gym.Env wrapper to stack multiple frames.
    Useful for training feed-forward agents on dynamic games.
    """

    def __init__(self, env, n_frames):
        """
        Stack frames wrapper.

        Args:
            env: gym.Env to wrap.
            n_frames: number of frames to stack.

        Raises:
            ValueError: If observation space shape is not 2.

        """
        super().__init__(env)
        if len(env.observation_space.shape) != 2:
            raise ValueError(
                "Stack frames only works with 2D single channel images")

        self.n_frames = n_frames
        self._frames = deque(maxlen=n_frames)

        new_obs_space_shape = env.observation_space.shape + (n_frames, )
        self.observation_space = Box(
            0.0, 1.0, shape=new_obs_space_shape, dtype=np.float32)

    def _stack_frames(self):
        return np.stack(self._frames, axis=2)

    def reset(self):
        """gym.Env reset function."""
        observation = self.env.reset()
        self._frames.clear()
        for i in range(self.n_frames):
            self._frames.append(observation)

        return self._stack_frames()

    def step(self, action):
        """gym.Env step function."""
        new_observation, reward, done, info = self.env.step(action)
        self._frames.append(new_observation)

        return self._stack_frames(), reward, done, info
