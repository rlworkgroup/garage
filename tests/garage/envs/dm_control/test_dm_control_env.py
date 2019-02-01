import collections
from copy import copy
import pickle
import unittest

from dm_control.suite import ALL_TASKS
from nose2.tools import params

from garage.envs.dm_control import DmControlEnv
from tests.helpers import step_env


class TestDmControlEnv(unittest.TestCase):
    @params(*ALL_TASKS)
    def test_can_step_and_render(self, domain_name, task_name):
        env = DmControlEnv(domain_name, task_name)
        ob_space = env.observation_space
        act_space = env.action_space
        ob = env.reset()
        assert ob_space.contains(ob)
        a = act_space.sample()
        assert act_space.contains(a)
        step_env(env, n=10, render=True)

    @params(*ALL_TASKS)
    def test_pickling(self, domain_name, task_name):
        env = DmControlEnv(domain_name, task_name)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        step_env(round_trip)

    @params(*ALL_TASKS)
    def test_all_does_not_modify_actions(self, domain_name, task_name):
        env = DmControlEnv(domain_name, task_name)
        a = env.action_space.sample()
        a_copy = copy(a)
        env.step(a)
        if isinstance(a, collections.Iterable):
            self.assertEquals(a.all(), a_copy.all())
        else:
            self.assertEquals(a, a_copy)
