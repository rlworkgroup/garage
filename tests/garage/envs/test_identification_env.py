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

    def test_does_not_modify_action(self):
        env = IdentificationEnv(CartpoleEnv, dict(obs_noise=5.))
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
