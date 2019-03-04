import unittest

import numpy as np

from garage.envs.wrappers import FireReset
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestFireReset(unittest.TestCase):
    def test_fire_reset(self):
        env = DummyDiscretePixelEnv()
        env_wrap = FireReset(env)
        obs = env.reset()
        obs_wrap = env_wrap.reset()

        assert np.array_equal(obs, np.zeros(env.observation_space.shape))
        assert np.array_equal(obs_wrap, np.ones(env.observation_space.shape))
