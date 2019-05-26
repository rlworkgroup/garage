import collections
from copy import copy
import pickle
import unittest

import dm_control.suite
from nose2.tools import params

from garage.envs.dm_control import DmControlEnv
from tests.helpers import step_env


class TestDmControlEnv(unittest.TestCase):
    def test_can_step(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DmControlEnv.from_suite(domain_name, task_name)
        ob_space = env.observation_space
        act_space = env.action_space
        ob = env.reset()
        assert ob_space.contains(ob)
        a = act_space.sample()
        assert act_space.contains(a)
        # Skip rendering because it causes TravisCI to run out of memory
        step_env(env, render=False)
        env.close()

    @params(*dm_control.suite.ALL_TASKS)
    def test_all_can_step(self, domain_name, task_name):
        env = DmControlEnv.from_suite(domain_name, task_name)
        ob_space = env.observation_space
        act_space = env.action_space
        ob = env.reset()
        assert ob_space.contains(ob)
        a = act_space.sample()
        assert act_space.contains(a)
        # Skip rendering because it causes TravisCI to run out of memory
        step_env(env, render=False)
        env.close()

    test_all_can_step.nightly = True

    def test_pickleable(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DmControlEnv.from_suite(domain_name, task_name)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        # Skip rendering because it causes TravisCI to run out of memory
        step_env(round_trip, render=False)
        round_trip.close()
        env.close()

    @params(*dm_control.suite.ALL_TASKS)
    def test_all_pickleable(self, domain_name, task_name):
        env = DmControlEnv.from_suite(domain_name, task_name)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        # Skip rendering because it causes TravisCI to run out of memory
        step_env(round_trip, render=False)
        round_trip.close()
        env.close()

    test_all_pickleable.nightly = True

    def test_does_not_modify_actions(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DmControlEnv.from_suite(domain_name, task_name)
        a = env.action_space.sample()
        a_copy = copy(a)
        env.step(a)
        if isinstance(a, collections.Iterable):
            self.assertEqual(a.all(), a_copy.all())
        else:
            self.assertEqual(a, a_copy)
        env.close()

    @params(*dm_control.suite.ALL_TASKS)
    def test_all_does_not_modify_actions(self, domain_name, task_name):
        env = DmControlEnv.from_suite(domain_name, task_name)
        a = env.action_space.sample()
        a_copy = copy(a)
        env.step(a)
        if isinstance(a, collections.Iterable):
            self.assertEqual(a.all(), a_copy.all())
        else:
            self.assertEqual(a, a_copy)
        env.close()

    test_all_does_not_modify_actions.nightly = True
