import pickle
import unittest

from garage.envs.mujoco.inverted_double_pendulum_env import (
    InvertedDoublePendulumEnv)
from tests.helpers import step_env


class TestInvertedDoublePendulumEnv(unittest.TestCase):
    def test_pickleable(self):
        env = InvertedDoublePendulumEnv(random_start=False)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.random_start == env.random_start
        step_env(round_trip)

    def test_does_not_modify_action(self):
        env = InvertedDoublePendulumEnv(random_start=False)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
