import pickle
import unittest

from garage.envs.box2d import CartpoleEnv
from garage.envs.occlusion_env import OcclusionEnv
from tests.helpers import step_env


class TestOcclusionEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = CartpoleEnv(obs_noise=5.)
        env = OcclusionEnv(inner_env, [1])
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        obs = inner_env.reset()
        assert round_trip.occlude(obs) == env.occlude(obs)
        assert round_trip.env.obs_noise == env.env.obs_noise
        step_env(round_trip)

    def test_does_not_modify_action(self):
        inner_env = CartpoleEnv(obs_noise=5.)
        env = OcclusionEnv(inner_env, [1])
        a = env.action_space.sample()
        a_copy = a
        env.reset()
        env.step(a)
        self.assertEquals(a, a_copy)
