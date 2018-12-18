import unittest
from unittest import mock

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

from garage.envs.wrappers import Grayscale
from garage.misc.overrides import overrides


class TestGrayscale(unittest.TestCase):
    @overrides
    def setUp(self):
        self.shape = (50, 50, 3)
        self.env = mock.Mock()
        self.env.observation_space = Box(
            low=0, high=255, shape=self.shape, dtype=np.uint8)
        self.env.reset.return_value = np.zeros(self.shape)
        self.env.step.side_effect = self._step

        self.env_g = Grayscale(self.env)

        self.obs = self.env.reset()
        self.obs_g = self.env_g.reset()

    def _step(self, action):
        return np.full(self.shape, 125), 0, False, dict()

    def test_gray_scale_invalid_environment_type(self):
        with self.assertRaises(ValueError):
            self.env.observation_space = Discrete(64)
            Grayscale(self.env)

    def test_gray_scale_invalid_environment_shape(self):
        with self.assertRaises(ValueError):
            self.env.observation_space = Box(
                low=0, high=255, shape=(4, ), dtype=np.uint8)
            Grayscale(self.env)

    def test_grayscale_observation_space(self):
        assert self.env_g.observation_space.shape == self.shape[:-1]

    def test_grayscale_reset(self):
        """
        RGB to grayscale conversion using scikit-image.

        Weights used for conversion:
        Y = 0.2125 R + 0.7154 G + 0.0721 B

        Reference:
        http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2grey
        """
        gray_scale_output = np.dot(self.obs[:, :, :3],
                                   [0.2125, 0.7154, 0.0721]) / 255.0
        np.testing.assert_array_almost_equal(gray_scale_output, self.obs_g)

    def test_grayscale_step(self):
        obs, _, _, _ = self.env.step(0)
        obs_g, _, _, _ = self.env_g.step(0)

        gray_scale_output = np.dot(obs[:, :, :3],
                                   [0.2125, 0.7154, 0.0721]) / 255.0
        np.testing.assert_array_almost_equal(gray_scale_output, obs_g)
