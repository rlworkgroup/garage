import pickle
import unittest

from garage.envs.mujoco.walker2d_env import Walker2DEnv
from tests.helpers import step_env


class TestWalker2DEnv(unittest.TestCase):
    def test_pickleable(self):
        env = Walker2DEnv(ctrl_cost_coeff=3.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.ctrl_cost_coeff == env.ctrl_cost_coeff
        step_env(round_trip)
