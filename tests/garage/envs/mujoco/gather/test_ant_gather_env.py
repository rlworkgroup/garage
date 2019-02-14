import pickle
import unittest

from garage.envs.mujoco.gather.ant_gather_env import AntGatherEnv
from tests.helpers import step_env


class TestAntGatherEnv(unittest.TestCase):
    def test_pickleable(self):
        env = AntGatherEnv(n_apples=1)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.n_apples == env.n_apples
        step_env(round_trip)
        round_trip.close()
        env.close()
