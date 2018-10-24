import pickle
import unittest

from garage.envs.mujoco.point_env import PointEnv
from tests.helpers import step_env


class TestPointEnv(unittest.TestCase):
    def test_pickleable(self):
        env = PointEnv(action_noise=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.action_noise == env.action_noise
        step_env(round_trip)

    def test_does_not_modify_action(self):
        env = PointEnv(action_noise=1.0)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
