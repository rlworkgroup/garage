import unittest

import gym

from garage.envs.wrappers import Resize


class TestResize(unittest.TestCase):
    def test_resize_output_shape(self):
        env_r = Resize(gym.make("Breakout-v0"), width=84, height=84)
        assert env_r.observation_space.shape == (84, 84)
