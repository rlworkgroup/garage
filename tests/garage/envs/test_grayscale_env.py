import unittest

import gym
import numpy as np

from garage.envs.wrappers import Grayscale


class TestGrayscale(unittest.TestCase):
    def test_gray_scale_invalid_environment(self):
        with self.assertRaises(ValueError):
            env = gym.make("FrozenLake8x8-v0")
            assert env.observation_space.n == 64

            Grayscale(env)

    def test_gray_scale_invalid_shape(self):
        with self.assertRaises(ValueError):
            env = gym.make("CartPole-v0")
            assert env.observation_space.shape == (4, )

            Grayscale(env)

    def test_gray_scale_output(self):
        """
        RGB to grayscale conversion using scikit-image.

        Weights used for conversion:
        Y = 0.2125 R + 0.7154 G + 0.0721 B

        Reference:
        http://scikit-image.org/docs/dev/api/skimage.color.html#skimage.color.rgb2grey
        """
        env = gym.make("Breakout-v0")
        self.env = env
        self.env_r = Grayscale(env)

        self.obs = self.env.reset()
        self.obs_r = self.env_r.reset()

        gray_scale_output = np.dot(self.obs[:, :, :3],
                                   [0.2125, 0.7154, 0.0721]) / 255.0
        np.testing.assert_array_almost_equal(gray_scale_output, self.obs_r)
