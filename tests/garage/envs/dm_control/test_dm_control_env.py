import pickle
import unittest

import dm_control
from nose2.tools import params

from garage.envs.dm_control.dm_control_env import DmControlEnv
from tests.helpers import step_env


class TestDmControlEnv(unittest.TestCase):
    @params(*dm_control.suite.ALL_TASKS)
    def test_all_pickleable(self, domain, task):
        env = DmControlEnv(domain, task)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        step_env(round_trip)

    @params(*dm_control.suite.ALL_TASKS)
    def test_all(self, domain, task):
        env = DmControlEnv(domain, task)
        step_env(env)
