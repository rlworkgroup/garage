import unittest
from unittest import mock

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

from garage.envs.wrappers import Resize
from garage.misc.overrides import overrides


class TestResize(unittest.TestCase):
    @overrides
    def setUp(self):
        self.shape = (50, 50)
        self.env = mock.Mock()
        self.env.observation_space = Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8)
        self.env.reset.return_value = np.zeros(self.shape)
        self.env.step.side_effect = self._step

        self._width = 16
        self._height = 16
        self.env_r = Resize(self.env, width=self._width, height=self._height)

        self.obs = self.env.reset()
        self.obs_r = self.env_r.reset()

    def _step(self, action):
        return np.full(self.shape, 125), 0, False, dict()

    def test_resize_invalid_environment_type(self):
        with self.assertRaises(ValueError):
            self.env.observation_space = Discrete(64)
            Resize(self.env, width=self._width, height=self._height)

    def test_resize_invalid_environment_shape(self):
        with self.assertRaises(ValueError):
            self.env.observation_space = Box(
                low=0, high=255, shape=(4, ), dtype=np.uint8)
            Resize(self.env, width=self._width, height=self._height)

    def test_resize_output_observation_space(self):
        assert self.env_r.observation_space.shape == (self._width,
                                                      self._height)

    def test_resize_output_reset(self):
        assert self.obs_r.shape == (self._width, self._height)

    def test_resize_output_step(self):
        obs_r, _, _, _ = self.env_r.step(0)

        assert obs_r.shape == (self._width, self._height)
