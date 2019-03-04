import unittest

from garage.envs.wrappers import Noop
from tests.fixtures.envs.dummy import DummyDiscretePixelEnv


class TestNoop(unittest.TestCase):
    def test_noop(self):
        env = Noop(DummyDiscretePixelEnv(), noop_max=3)

        for _ in range(1000):
            env.reset()
            assert 1 <= env.env.step_called <= 3
