import pickle
import unittest

from garage.envs.box2d import CartpoleEnv
from garage.envs.identification_env import IdentificationEnv
from tests.helpers import step_env


class TestIdentificationEnv(unittest.TestCase):
    def test_pickleable(self):
        env = IdentificationEnv(CartpoleEnv, dict(obs_noise=5.))
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert env.mdp_args == round_trip.mdp_args
        assert round_trip.env.obs_noise == env.env.obs_noise
        step_env(round_trip)
