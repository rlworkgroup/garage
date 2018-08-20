import pickle
import unittest

from garage.envs.mujoco.humanoid_env import HumanoidEnv
from tests.helpers import step_env


class TestHumanoidEnv(unittest.TestCase):
    def test_pickleable(self):
        env = HumanoidEnv(alive_bonus=1.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.alive_bonus == env.alive_bonus
        step_env(round_trip)
