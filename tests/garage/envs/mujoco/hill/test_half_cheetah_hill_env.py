import pickle
import unittest

from garage.envs.mujoco.hill.half_cheetah_hill_env import HalfCheetahHillEnv
from tests.helpers import step_env


class TestHalfCheetahHillEnv(unittest.TestCase):
    def test_pickleable(self):
        env = HalfCheetahHillEnv(regen_terrain=False)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.difficulty == env.difficulty
        step_env(round_trip)
