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
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = CarParkingEnv(random_start_range=2.0)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
        env.close()
