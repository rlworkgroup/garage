import unittest

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

from garage.envs.wrappers import Grayscale
from garage.tf.envs import TfEnv
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestGrayscale(unittest.TestCase):
    def setUp(self):
        self.env = TfEnv(DummyDiscretePixelEnv(random=False))
        self.env_g = TfEnv(Grayscale(DummyDiscretePixelEnv(random=False)))

    def tearDown(self):
        self.env.close()
        self.env_g.close()

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
        assert self.env_g.observation_space.shape == (
            self.env.observation_space.shape[:-1])

    def test_grayscale_reset(self):
        """
        RGB to grayscale conversion using scikit-image.

        Weights used for conversion:
        Y = 0.2125 R + 0.7154 G + 0.0721 B

        Reference:
        http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2grey
        """
        gray_scale_output = np.round(
            np.dot(self.env.reset()[:, :, :3],
                   [0.2125, 0.7154, 0.0721])).astype(np.uint8)
        np.testing.assert_array_almost_equal(gray_scale_output,
                                             self.env_g.reset())

    def test_grayscale_step(self):
        self.env.reset()
        self.env_g.reset()
        obs, _, _, _ = self.env.step(1)
        obs_g, _, _, _ = self.env_g.step(1)

        gray_scale_output = np.round(
            np.dot(obs[:, :, :3], [0.2125, 0.7154, 0.0721])).astype(np.uint8)
        np.testing.assert_array_almost_equal(gray_scale_output, obs_g)
