import unittest

from gym.spaces import Box
from gym.spaces import Discrete
import numpy as np

from garage.envs.wrappers import Resize
from garage.tf.envs import TfEnv
from tests.fixtures.envs.dummy import DummyDiscrete2DEnv


class TestResize(unittest.TestCase):
    def setUp(self):
        self.width = 16
        self.height = 16
        self.env = TfEnv(DummyDiscrete2DEnv())
        self.env_r = TfEnv(
            Resize(DummyDiscrete2DEnv(), width=self.width, height=self.height))

    def tearDown(self):
        self.env.close()
        self.env_r.close()

    def test_resize_invalid_environment_type(self):
        with self.assertRaises(ValueError):
            self.env.observation_space = Discrete(64)
            Resize(self.env, width=self.width, height=self.height)

    def test_resize_invalid_environment_shape(self):
        with self.assertRaises(ValueError):
            self.env.observation_space = Box(
                low=0, high=255, shape=(4, ), dtype=np.uint8)
            Resize(self.env, width=self.width, height=self.height)

    def test_resize_output_observation_space(self):
        assert self.env_r.observation_space.shape == (self.width, self.height)

    def test_resize_output_reset(self):
        assert self.env_r.reset().shape == (self.width, self.height)

    def test_resize_output_step(self):
        self.env_r.reset()
        obs_r, _, _, _ = self.env_r.step(1)
        assert obs_r.shape == (self.width, self.height)
