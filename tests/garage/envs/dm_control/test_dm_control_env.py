import collections
from copy import copy
import pickle

import dm_control.suite
import pytest

from garage.envs.dm_control import DmControlEnv
from tests.helpers import step_env


class TestDmControlEnv:
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

    @pytest.mark.nightly
    @pytest.mark.parametrize('domain_name, task_name',
                             dm_control.suite.ALL_TASKS)
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

    def test_pickleable(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DmControlEnv.from_suite(domain_name, task_name)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        # Skip rendering because it causes TravisCI to run out of memory
        step_env(round_trip, render=False)
        round_trip.close()
        env.close()

    @pytest.mark.nightly
    @pytest.mark.parametrize('domain_name, task_name',
                             dm_control.suite.ALL_TASKS)
    def test_all_pickleable(self, domain_name, task_name):
        env = DmControlEnv.from_suite(domain_name, task_name)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        # Skip rendering because it causes TravisCI to run out of memory
        step_env(round_trip, render=False)
        round_trip.close()
        env.close()

    def test_does_not_modify_actions(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DmControlEnv.from_suite(domain_name, task_name)
        a = env.action_space.sample()
        a_copy = copy(a)
        env.step(a)
        if isinstance(a, collections.Iterable):
            assert a.all() == a_copy.all()
        else:
            assert a == a_copy
        env.close()

    @pytest.mark.nightly
    @pytest.mark.parametrize('domain_name, task_name',
                             dm_control.suite.ALL_TASKS)
    def test_all_does_not_modify_actions(self, domain_name, task_name):
        env = DmControlEnv.from_suite(domain_name, task_name)
        a = env.action_space.sample()
        a_copy = copy(a)
        env.step(a)
        if isinstance(a, collections.Iterable):
            assert a.all() == a_copy.all()
        else:
            assert a == a_copy
        env.close()
