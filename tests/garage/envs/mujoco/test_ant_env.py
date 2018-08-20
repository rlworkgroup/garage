import pickle
import unittest

from garage.envs.mujoco.ant_env import AntEnv
from tests.helpers import step_env


class TestAntEnv(unittest.TestCase):
    def test_pickleable(self):
        env = AntEnv(action_noise=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.action_noise == env.action_noise
        step_env(round_trip)
