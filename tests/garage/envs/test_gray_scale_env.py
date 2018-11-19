import unittest

import gym
import numpy as np

from garage.envs.wrappers import GrayScale
from garage.misc.overrides import overrides


class TestGrayScale(unittest.TestCase):
    @overrides
    def setUp(self):
        self.env = gym.make("Breakout-v0")
        self.env_r = GrayScale(gym.make("Breakout-v0"))

        self.obs = self.env.reset()
        self.obs_r = self.env_r.reset()

    def test_gray_scale_output(self):

        gray_color = np.dot(self.obs[:, :, :3], [0.299, 0.587, 0.114]) / 255.0
        np.testing.assert_array_equal(gray_color, self.obs_r)
