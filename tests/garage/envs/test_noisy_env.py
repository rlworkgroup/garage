import pickle
import unittest

import numpy as np

from garage.envs import DelayedActionEnv
from garage.envs import NoisyObservationEnv
from garage.envs import PointEnv
from tests.helpers import step_env


class TestDelayedActionEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = DelayedActionEnv(inner_env, action_delay=10)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.action_delay == env.action_delay
        assert np.array_equal(round_trip.env._goal, env.env._goal)
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = DelayedActionEnv(inner_env, action_delay=10)
        env.reset()
        a = env.action_space.high + 1.
        a_copy = a.copy()
        env.reset()
        env.step(a)
        assert np.array_equal(a, a_copy)
        env.close()


class TestNoisyObservationEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = NoisyObservationEnv(inner_env, obs_noise=5.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.obs_noise == env.obs_noise
        assert np.array_equal(round_trip.env._goal, env.env._goal)
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = NoisyObservationEnv(inner_env, obs_noise=5.)
        a = env.action_space.high + 1.
        a_copy = a.copy()
        env.step(a)
        assert np.array_equal(a, a_copy)
        env.close()
