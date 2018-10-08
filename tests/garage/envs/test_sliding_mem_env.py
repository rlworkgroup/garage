import pickle
import unittest

from garage.envs.box2d import CartpoleEnv
from garage.envs.sliding_mem_env import SlidingMemEnv
from tests.helpers import step_env


class TestSlidingMemEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = CartpoleEnv(obs_noise=5.)
        env = SlidingMemEnv(inner_env, n_steps=10)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.n_steps == env.n_steps
        assert round_trip.env.obs_noise == env.env.obs_noise
        step_env(round_trip)
