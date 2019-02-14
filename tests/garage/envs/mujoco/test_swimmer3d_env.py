import pickle
import unittest

from garage.envs.mujoco.swimmer3d_env import Swimmer3DEnv
from tests.helpers import step_env


class TestSwimmer3DEnv(unittest.TestCase):
    def test_pickleable(self):
        env = Swimmer3DEnv(ctrl_cost_coeff=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.ctrl_cost_coeff == env.ctrl_cost_coeff
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = Swimmer3DEnv(ctrl_cost_coeff=1.0)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
        env.close()
