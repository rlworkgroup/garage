import pickle
import unittest

from garage.envs import DelayedActionEnv
from garage.envs import NoisyObservationEnv
from garage.envs.box2d import CartpoleEnv
from tests.helpers import step_env


class TestDelayedActionEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = CartpoleEnv(frame_skip=10)
        env = DelayedActionEnv(inner_env, action_delay=10)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.action_delay == env.action_delay
        assert round_trip.env.frame_skip == env.env.frame_skip
        step_env(round_trip)

    def test_does_not_modify_action(self):
        inner_env = CartpoleEnv(frame_skip=10)
        env = DelayedActionEnv(inner_env, action_delay=10)
        env.reset()
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())


class TestNoisyObservationEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = CartpoleEnv(frame_skip=10)
        env = NoisyObservationEnv(inner_env, obs_noise=5.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.obs_noise == env.obs_noise
        assert round_trip.env.frame_skip == env.env.frame_skip
        step_env(round_trip)

    def test_does_not_modify_action(self):
        inner_env = CartpoleEnv(frame_skip=10)
        env = NoisyObservationEnv(inner_env, obs_noise=5.)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
