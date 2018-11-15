"""
Gym env wrapper to stack multiple frames.
Useful for training feed-forward agents on dynamic games.
"""
import gym
import numpy as np

from collections import deque
from gym.spaces import Box


class StackFramesWrapper(gym.core.Wrapper):
    def __init__(self, env, n_frames_stacked):
        super(StackFramesWrapper, self).__init__(env)
        if len(env.observation_space.shape) != 2:
            raise Exception('Stack frames works with 2D single channel images')
        self._n_stack_past = n_frames_stacked
        self._frames = None

        new_obs_space_shape = env.observation_space.shape + (
            n_frames_stacked, )
        self.observation_space = Box(
            0.0, 1.0, shape=new_obs_space_shape, dtype=np.float32)

    def _frames_as_numpy(self):
        np_frames = np.asarray(self._frames)
        np_frames = np.transpose(np_frames, axes=[1, 2, 0])
        return np_frames

    def reset(self):
        observation = self.env.reset()
        self._frames = deque([observation] * self._n_stack_past)
        return self._frames_as_numpy()

    def step(self, action):
        new_observation, reward, done, info = self.env.step(action)
        self._frames.popleft()
        self._frames.append(new_observation)
        return self._frames_as_numpy(), reward, done, info
