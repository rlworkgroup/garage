import pickle
import unittest

from garage.envs.mujoco.humanoid_env import HumanoidEnv
from tests.helpers import step_env


class TestHumanoidEnv(unittest.TestCase):
    def test_pickleable(self):
        env = HumanoidEnv(alive_bonus=1.)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert round_trip.alive_bonus == env.alive_bonus
        step_env(round_trip)

    def test_does_not_modify_action(self):
        env = HumanoidEnv(alive_bonus=1.)
        a = env.action_space.sample()
        a_copy = a.copy()
        env.reset()
        env.step(a)
        self.assertEquals(a.all(), a_copy.all())
