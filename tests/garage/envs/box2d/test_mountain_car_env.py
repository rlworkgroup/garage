import pickle
import unittest

from garage.envs.box2d.mountain_car_env import MountainCarEnv
from tests.helpers import step_env


class TestMountainCarEnv(unittest.TestCase):
    def test_pickleable(self):
        env = MountainCarEnv(height_bonus=10.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.height_bonus == env.height_bonus
        step_env(round_trip)
