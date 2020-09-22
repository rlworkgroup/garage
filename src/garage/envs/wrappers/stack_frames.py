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
        env (gym.Env): gym.Env to wrap.
        n_frames (int): number of frames to stack.
        axis (int): Axis to stack frames on. This should be 2 for tensorflow
            and 0 for pytorch.

    Raises:
         ValueError: If observation space shape is not 2 dimnesional,
         if the environment is not gym.spaces.Box, or if the specified axis
         is not 0 or 2.


    """

    def __init__(self, env, n_frames, axis=2):
        if axis not in (0, 2):
            raise ValueError('Frame stacking axis should be 0 for pytorch or '
                             '2 for tensorflow.')
        if not isinstance(env.observation_space, gym.spaces.Box):
            raise ValueError('Stack frames only works with gym.spaces.Box '
                             'environment.')

        if len(env.observation_space.shape) != 2:
            raise ValueError(
                'Stack frames only works with 2D single channel images')

        super().__init__(env)

        self._n_frames = n_frames
        self._axis = axis
        self._frames = deque(maxlen=n_frames)

        new_obs_space_shape = env.observation_space.shape + (n_frames, )
        if axis == 0:
            new_obs_space_shape = (n_frames, ) + env.observation_space.shape

        _low = env.observation_space.low.flatten()[0]
        _high = env.observation_space.high.flatten()[0]
        self._observation_space = gym.spaces.Box(
            _low,
            _high,
            shape=new_obs_space_shape,
            dtype=env.observation_space.dtype)

    @property
    def observation_space(self):
        """gym.spaces.Box: gym.Env observation space."""
        return self._observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self._observation_space = observation_space

    def _stack_frames(self):
        """Stacks and returns the last n_frames.

        Returns:
            np.ndarray: stacked observation with shape either
            :math:`(N, n_frames, O*)` or :math:(N, O*, n_frames),
            depending on the axis specified.
        """
        return np.stack(self._frames, axis=self._axis)

    # pylint: disable=arguments-differ
    def reset(self):
        """gym.Env reset function.

        Returns:
            np.ndarray: Observation conforming to observation_space
            float: Reward for this step
            bool: Termination signal
            dict: Extra information from the environment.
        """
        observation = self.env.reset()
        self._frames.clear()
        for _ in range(self._n_frames):
            self._frames.append(observation)

        return self._stack_frames()

    def step(self, action):
        """gym.Env step function.

        Args:
            action (int): index of the action to take.

        Returns:
            np.ndarray: Observation conforming to observation_space
            float: Reward for this step
            bool: Termination signal
            dict: Extra information from the environment.
        """
        new_observation, reward, done, info = self.env.step(action)
        self._frames.append(new_observation)

        return self._stack_frames(), reward, done, info
