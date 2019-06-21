import numpy as np

from garage.envs.wrappers import Noop
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestNoop:
    def test_noop(self):
        env = Noop(DummyDiscretePixelEnv(), noop_max=3)

        for _ in range(1000):
            env.reset()
            assert 1 <= env.env.step_called <= 3

        env = Noop(DummyDiscretePixelEnv(), noop_max=10)
        for _ in range(1000):
            obs = env.reset()
            if env.env.step_called % 5 == 0:
                """
                There are only 5 lives in the environment, so if number of
                steps are multiple of 5, env will call reset at last.
                """
                assert np.array_equal(obs,
                                      np.ones(env.observation_space.shape))
            else:
                assert not np.array_equal(obs,
                                          np.ones(env.observation_space.shape))
