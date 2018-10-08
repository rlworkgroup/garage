import pickle
import unittest

from garage.envs.mujoco.gather.swimmer_gather_env import SwimmerGatherEnv
from tests.helpers import step_env


class TestSwimmerGatherEnv(unittest.TestCase):
    def test_pickleable(self):
        env = SwimmerGatherEnv(n_apples=1)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.n_apples == env.n_apples
        step_env(round_trip)
