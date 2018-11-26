import unittest

import gym
import numpy as np

from garage.envs.wrappers import GrayScale
from garage.envs.wrappers import StackFrames
from garage.misc.overrides import overrides


class TestStackFrames(unittest.TestCase):
    @overrides
    def setUp(self):
        self.env_stack = StackFrames(
            GrayScale(gym.make("Breakout-v0")), n_frames=4)
        self.env = GrayScale(gym.make("Breakout-v0"))

        self.env.seed(0)
        self.env_stack.seed(0)
        self.obs = self.env.reset()
        self.obs_stack = self.env_stack.reset()
        self.frame_width = self.env.observation_space.shape[0]
        self.frame_height = self.env.observation_space.shape[1]

    def test_stack_frames_multiple_channel(self):
        with self.assertRaises(ValueError):
            StackFrames(gym.make("Breakout-v0"), n_frames=4)

    def test_stack_frames_for_reset(self):
        frame_stack = self.obs
        for i in range(3):
            frame_stack = np.dstack((frame_stack, self.obs))

        assert self.obs_stack.shape == (self.frame_width, self.frame_height, 4)

        np.testing.assert_array_equal(self.obs_stack, frame_stack)

    def test_stack_frames_for_step(self):
        frame_stack = np.empty((self.frame_width, self.frame_height, 4))
        for i in range(10):
            frame_stack = frame_stack[:, :, 1:]
            obs, _, _, _ = self.env.step(1)
            frame_stack = np.dstack((frame_stack, obs))
            obs_stack, _, _, _ = self.env_stack.step(1)
        np.testing.assert_array_equal(obs_stack, frame_stack)
