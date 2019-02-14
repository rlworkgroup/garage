import pickle
import unittest

from garage.envs.mujoco.gather.point_gather_env import PointGatherEnv
from tests.helpers import step_env


class TestPointGatherEnv(unittest.TestCase):
    def test_pickleable(self):
        env = PointGatherEnv(n_apples=1)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.n_apples == env.n_apples
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = PointGatherEnv(n_apples=1)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
        env.close()
