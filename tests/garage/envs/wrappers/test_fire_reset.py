import numpy as np

from garage.envs.wrappers import FireReset
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestFireReset:
    def test_fire_reset(self):
        env = DummyDiscretePixelEnv()
        env_wrap = FireReset(env)
        obs = env.reset()
        obs_wrap = env_wrap.reset()

        assert np.array_equal(obs, np.ones(env.observation_space.shape))
        assert np.array_equal(obs_wrap, np.full(env.observation_space.shape,
                                                2))

        env_wrap.step(2)
        obs_wrap = env_wrap.reset()  # env will call reset again, after fire
        assert np.array_equal(obs_wrap, np.ones(env.observation_space.shape))
