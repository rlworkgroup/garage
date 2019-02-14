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
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = HopperEnv(alive_coeff=2.0)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
        env.close()
