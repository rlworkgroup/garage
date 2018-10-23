import pickle
import unittest

from garage.envs.grid_world_env import GridWorldEnv
from tests.helpers import step_env


class TestGridWorldEnv(unittest.TestCase):
    def test_pickleable(self):
        env = GridWorldEnv(desc="8x8")
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.start_state == env.start_state
        step_env(round_trip)

    def test_does_not_modify_action(self):
        env = GridWorldEnv(desc="8x8")
        a = env.action_space.sample()
        a_copy = a
        env.reset()
        env.step(a)
        self.assertEquals(a, a_copy)
