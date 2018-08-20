import pickle
import unittest

from garage.envs.mujoco.hill.hopper_hill_env import HopperHillEnv
from tests.helpers import step_env


class TestHopperHillEnv(unittest.TestCase):
    def test_pickleable(self):
        env = HopperHillEnv(regen_terrain=False)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.difficulty == env.difficulty
        step_env(round_trip)
