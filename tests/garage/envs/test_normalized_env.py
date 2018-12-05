import pickle
import unittest

from garage.envs.box2d import CartpoleEnv
from garage.envs.mujoco import SwimmerEnv
from garage.envs.normalized_env import normalize, NormalizedEnv
from tests.helpers import step_env


class TestNormalizedEnv(unittest.TestCase):
    def test_can_create_env(self):
        # Fixes https://github.com/rlworkgroup/garage/pull/420
        env = normalize(SwimmerEnv())
        assert env

    def test_pickleable(self):
        inner_env = CartpoleEnv(obs_noise=5.)
        env = NormalizedEnv(inner_env, scale_reward=10.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip._scale_reward == env._scale_reward
        assert round_trip.env.obs_noise == env.env.obs_noise
        step_env(round_trip)

    def test_does_not_modify_action(self):
        inner_env = CartpoleEnv(obs_noise=5.)
        env = NormalizedEnv(inner_env, scale_reward=10.)
        a = env.action_space.sample()
        a_copy = a
        env.reset()
        env.step(a)
        self.assertEquals(a, a_copy)
