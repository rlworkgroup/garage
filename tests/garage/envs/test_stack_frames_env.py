import unittest
from unittest import mock

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

from garage.envs.wrappers import StackFrames
from garage.misc.overrides import overrides


class TestStackFrames(unittest.TestCase):
    @overrides
    def setUp(self):
        self.shape = (50, 50)
        self.env = mock.Mock()
        self.env.observation_space = Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8)
        self.env.reset.return_value = np.zeros(self.shape)
        self.env.step.side_effect = self._step

        self._n_frames = 4
        self.env_s = StackFrames(self.env, n_frames=self._n_frames)

        self.obs = self.env.reset()
        self.obs_s = self.env_s.reset()
        self.frame_width = self.env.observation_space.shape[0]
        self.frame_height = self.env.observation_space.shape[1]

    def _step(self, action):
        def generate():
            for i in range(0, 255):
                yield np.full(self.shape, i)

        generator = generate()

        return next(generator), 0, False, dict()

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
        assert self.env_s.observation_space.shape == (self.frame_width,
                                                      self.frame_height,
                                                      self._n_frames)

    def test_stack_frames_for_reset(self):
        frame_stack = self.obs
        for i in range(self._n_frames - 1):
            frame_stack = np.dstack((frame_stack, self.obs))

        np.testing.assert_array_equal(self.obs_s, frame_stack)

    def test_stack_frames_for_step(self):
        frame_stack = np.empty((self.frame_width, self.frame_height,
                                self._n_frames))
        for i in range(10):
            frame_stack = frame_stack[:, :, 1:]
            obs, _, _, _ = self.env.step(0)
            frame_stack = np.dstack((frame_stack, obs))

        obs_stack, _, _, _ = self.env_s.step(0)
        np.testing.assert_array_equal(obs_stack, frame_stack)
