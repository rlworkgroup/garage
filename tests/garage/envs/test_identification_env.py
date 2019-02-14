import pickle
import unittest

import numpy as np

from garage.envs import PointEnv
from garage.envs.identification_env import IdentificationEnv
from tests.helpers import step_env


class TestIdentificationEnv(unittest.TestCase):
    def test_pickleable(self):
        env = IdentificationEnv(PointEnv, dict(goal=(1., 2.)))
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        assert env.mdp_args == round_trip.mdp_args
        assert np.array_equal(round_trip.env._goal, env.env._goal)
        step_env(round_trip)
        round_trip.close()
        env.close()

    def test_does_not_modify_action(self):
        env = IdentificationEnv(PointEnv, dict(goal=(1., 2.)))
        a = env.action_space.high + 1.
        a_copy = a.copy()
        env.reset()
        env.step(a)
        assert np.array_equal(a, a_copy)
        env.close()
