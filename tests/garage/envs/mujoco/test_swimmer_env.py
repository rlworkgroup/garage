import pickle
import unittest

from garage.envs.mujoco.swimmer_env import SwimmerEnv
from tests.helpers import step_env


class TestSwimmerEnv(unittest.TestCase):
    def test_pickleable(self):
        env = SwimmerEnv(ctrl_cost_coeff=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.ctrl_cost_coeff == env.ctrl_cost_coeff
        step_env(round_trip)
