import unittest

import numpy as np

from garage.envs.wrappers import RepeatAction
from garage.tf.envs import TfEnv
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestRepeatAction(unittest.TestCase):
    def setUp(self):
        self.env = TfEnv(DummyDiscreteEnv(random=False))
        self.env_r = TfEnv(
            RepeatAction(DummyDiscreteEnv(random=False), n_frame_to_repeat=4))

    def tearDown(self):
        self.env.close()
        self.env_r.close()

    def test_repeat_action_reset(self):
        np.testing.assert_array_equal(self.env.reset(), self.env_r.reset())

    def test_repeat_action_step(self):
        self.env.reset()
        self.env_r.reset()
        obs_repeat, _, _, _ = self.env_r.step(1)
        for i in range(4):
            obs, _, _, _ = self.env.step(1)

        np.testing.assert_array_equal(obs, obs_repeat)
