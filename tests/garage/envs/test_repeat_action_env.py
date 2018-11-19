import unittest

import gym
import numpy as np

from garage.envs.wrappers import RepeatAction
from garage.misc.overrides import overrides


class TestRepeatAction(unittest.TestCase):
    @overrides
    def setUp(self):
        self.env_repeat = RepeatAction(
            gym.make("Breakout-v0"), n_frame_to_repeat=4)
        self.env = gym.make("Breakout-v0")

        self.env.seed(0)
        self.env_repeat.seed(0)
        self.env.reset()
        self.env_repeat.reset()

    def test_repeat_action_output(self):
        obs_repeat, _, _, _ = self.env_repeat.step(1)
        for i in range(4):
            obs, _, _, _ = self.env.step(1)

        np.testing.assert_array_equal(obs, obs_repeat)
