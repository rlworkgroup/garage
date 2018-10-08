import pickle
import unittest

from garage.envs.box2d.cartpole_env import CartpoleEnv
from tests.helpers import step_env


class TestCartpoleEnv(unittest.TestCase):
    def test_pickleable(self):
        env = CartpoleEnv(obs_noise=1.0)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.obs_noise == env.obs_noise
        step_env(round_trip)
