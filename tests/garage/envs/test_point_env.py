import pickle
import unittest

from garage.envs.point_env import PointEnv
from tests.helpers import step_env


class TestPointEnv(unittest.TestCase):
    def test_pickleable(self):
        env = PointEnv()
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        step_env(round_trip)

    def test_does_not_modify_action(self):
        env = PointEnv()
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
