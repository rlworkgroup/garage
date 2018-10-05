import pickle
import unittest

from garage.envs.box2d.car_parking_env import CarParkingEnv
from tests.helpers import step_env


class TestCarParkingEnv(unittest.TestCase):
    def test_pickleable(self):
        env = CarParkingEnv(random_start_range=2.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.random_start_range == env.random_start_range
        step_env(round_trip)
