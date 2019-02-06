import pickle
import unittest

import numpy as np

from garage.envs import PointEnv
from garage.envs.sliding_mem_env import SlidingMemEnv
from tests.helpers import step_env


class TestSlidingMemEnv(unittest.TestCase):
    def test_pickleable(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = SlidingMemEnv(inner_env, n_steps=10)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.n_steps == env.n_steps
        assert np.array_equal(round_trip.env._goal, env.env._goal)
        step_env(round_trip)

    def test_does_not_modify_action(self):
        inner_env = PointEnv(goal=(1., 2.))
        env = SlidingMemEnv(inner_env, n_steps=10)
        a = env.action_space.high + 1.
        a_copy = a.copy()
        env.reset()
        env.step(a)
        assert np.array_equal(a, a_copy)
