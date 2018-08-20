import pickle
import unittest

from garage.envs.mujoco.hill.walker2d_hill_env import Walker2DHillEnv
from tests.helpers import step_env


class TestWalker2DHillEnv(unittest.TestCase):
    def test_pickleable(self):
        env = Walker2DHillEnv(regen_terrain=False)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.difficulty == env.difficulty
        step_env(round_trip)
