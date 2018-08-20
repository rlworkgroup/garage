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
