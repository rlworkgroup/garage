import pickle
import unittest

from garage.envs.mujoco.simple_humanoid_env import SimpleHumanoidEnv
from tests.helpers import step_env


class TestSimpleHumanoidEnv(unittest.TestCase):
    def test_pickleable(self):
        env = SimpleHumanoidEnv(alive_bonus=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.alive_bonus == env.alive_bonus
        step_env(round_trip)
