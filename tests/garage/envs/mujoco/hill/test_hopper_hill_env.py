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

    def test_does_not_modify_action(self):
        env = HopperHillEnv(regen_terrain=False)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
