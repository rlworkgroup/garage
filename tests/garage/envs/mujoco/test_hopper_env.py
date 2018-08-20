import pickle
import unittest

from garage.envs.mujoco.hopper_env import HopperEnv
from tests.helpers import step_env


class TestHopperEnv(unittest.TestCase):
    def test_pickleable(self):
        env = HopperEnv(alive_coeff=2.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.alive_coeff == env.alive_coeff
        step_env(round_trip)
