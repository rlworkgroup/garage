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
