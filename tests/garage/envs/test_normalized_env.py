import pickle
import unittest

from garage.envs.box2d import CartpoleEnv
from garage.envs.normalized_env import NormalizedEnv
from tests.helpers import step_env


class TestNormalizedEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = CartpoleEnv(obs_noise=5.)
        env = NormalizedEnv(inner_env, scale_reward=10.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip._scale_reward == env._scale_reward
        assert round_trip.env.obs_noise == env.env.obs_noise
        step_env(round_trip)
