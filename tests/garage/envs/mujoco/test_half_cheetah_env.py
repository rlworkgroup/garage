import pickle
import unittest

from garage.envs.mujoco.half_cheetah_env import HalfCheetahEnv
from tests.helpers import step_env


class TestHalfCheetahEnv(unittest.TestCase):
    def test_pickleable(self):
        env = HalfCheetahEnv(action_noise=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.action_noise == env.action_noise
        step_env(round_trip)
