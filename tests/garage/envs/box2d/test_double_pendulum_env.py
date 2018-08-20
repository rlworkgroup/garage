import pickle
import unittest

from garage.envs.box2d.double_pendulum_env import DoublePendulumEnv
from tests.helpers import step_env


class TestDoublePendulumEnv(unittest.TestCase):
    def test_pickleable(self):
        env = DoublePendulumEnv(obs_noise=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.obs_noise == env.obs_noise
        step_env(round_trip)
