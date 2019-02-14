import pickle
import unittest

import numpy as np

from garage.envs import PointEnv
from garage.envs.occlusion_env import OcclusionEnv
from tests.helpers import step_env


class TestOcclusionEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = PointEnv(goal=(1, 2))
        env = OcclusionEnv(inner_env, [1])
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        obs = inner_env.reset()
        assert round_trip.occlude(obs) == env.occlude(obs)
        assert np.array_equal(round_trip.env._goal, env.env._goal)
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        inner_env = PointEnv(goal=(2, 3))
        env = OcclusionEnv(inner_env, [1])
        a = env.action_space.high + 1.
        a_copy = a
        env.reset()
        env.step(a)
        assert np.array_equal(a, a_copy)
        env.close()
