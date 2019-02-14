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
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = AntHillEnv(regen_terrain=False)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
        env.close()
