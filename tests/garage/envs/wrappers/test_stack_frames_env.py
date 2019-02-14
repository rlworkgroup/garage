import unittest

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

from garage.envs.wrappers import StackFrames
from garage.tf.envs import TfEnv
from tests.fixtures.envs.dummy import DummyDiscrete2DEnv


class TestStackFrames(unittest.TestCase):
    def setUp(self):
        self.n_frames = 4
        self.env = TfEnv(DummyDiscrete2DEnv(random=False))
        self.env_s = TfEnv(
            StackFrames(
                DummyDiscrete2DEnv(random=False), n_frames=self.n_frames))
        self.width, self.height = self.env.observation_space.shape

    def tearDown(self):
        self.env.close()
        self.env_s.close()

    def test_stack_frames_invalid_environment_type(self):
        with self.assertRaises(ValueError):
            self.env.observation_space = Discrete(64)
            StackFrames(self.env, n_frames=4)

    def test_stack_frames_invalid_environment_shape(self):
        with self.assertRaises(ValueError):
            self.env.observation_space = Box(
                low=0, high=255, shape=(4, ), dtype=np.uint8)
            StackFrames(self.env, n_frames=4)

    def test_stack_frames_output_observation_space(self):
        assert self.env_s.observation_space.shape == (self.width, self.height,
                                                      self.n_frames)

    def test_stack_frames_for_reset(self):
        frame_stack = self.env.reset()
        for i in range(self.n_frames - 1):
            frame_stack = np.dstack((frame_stack, self.env.reset()))

        np.testing.assert_array_equal(self.env_s.reset(), frame_stack)

    def test_stack_frames_for_step(self):
        self.env.reset()
        self.env_s.reset()

        frame_stack = np.empty((self.width, self.height, self.n_frames))
        for i in range(10):
            frame_stack = frame_stack[:, :, 1:]
            obs, _, _, _ = self.env.step(1)
            frame_stack = np.dstack((frame_stack, obs))
            obs_stack, _, _, _ = self.env_s.step(1)

        np.testing.assert_array_equal(obs_stack, frame_stack)
