import pickle
import unittest

from garage.envs.mujoco.hill.ant_hill_env import AntHillEnv
from tests.helpers import step_env


class TestAntHillEnv(unittest.TestCase):
    def test_pickleable(self):
        env = AntHillEnv(regen_terrain=False)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.difficulty == env.difficulty
        step_env(round_trip)
