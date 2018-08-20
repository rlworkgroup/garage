import pickle
import unittest

from garage.envs.mujoco.hill.swimmer3d_hill_env import Swimmer3DHillEnv
from tests.helpers import step_env


class TestSwimmer3DHillEnv(unittest.TestCase):
    def test_pickleable(self):

        env = Swimmer3DHillEnv(regen_terrain=False)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.difficulty == env.difficulty
        step_env(round_trip)
