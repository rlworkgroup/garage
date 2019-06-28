import pickle

import numpy as np

from garage.envs import PointEnv
from garage.envs.normalized_env import NormalizedEnv
from tests.helpers import step_env


class TestNormalizedEnv:
    def test_pickleable(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = NormalizedEnv(inner_env, scale_reward=10.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip._scale_reward == env._scale_reward
        assert np.array_equal(round_trip.env._goal, env.env._goal)
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = NormalizedEnv(inner_env, scale_reward=10.)
        a = env.action_space.high + 1.
        a_copy = a
        env.reset()
        env.step(a)
        assert np.array_equal(a, a_copy)
        env.close()
