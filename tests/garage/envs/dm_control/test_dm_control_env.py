import collections
from copy import copy
import pickle

import dm_control.mujoco
import dm_control.suite
import pytest

from garage.envs.dm_control import DMControlEnv

from tests.helpers import step_env


@pytest.mark.mujoco
class TestDMControlEnv:

    def test_can_step(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DMControlEnv.from_suite(domain_name, task_name)
        ob_space = env.observation_space
        act_space = env.action_space
        ob, _ = env.reset()
        assert ob_space.contains(ob)
        a = act_space.sample()
        assert act_space.contains(a)
        # Skip rendering because it causes TravisCI to run out of memory
        # Sometimes random actions lead to physics errors
        with env._env.physics.suppress_physics_errors():
            step_env(env, visualize=False)
        env.close()

    @pytest.mark.nightly
    @pytest.mark.parametrize('domain_name, task_name',
                             dm_control.suite.ALL_TASKS)
    def test_all_can_step(self, domain_name, task_name):
        env = DMControlEnv.from_suite(domain_name, task_name)
        ob_space = env.observation_space
        act_space = env.action_space
        ob, _ = env.reset()
        assert ob_space.contains(ob)
        a = act_space.sample()
        assert act_space.contains(a)
        # Skip rendering because it causes TravisCI to run out of memory
        # Sometimes random actions lead to physics errors
        with env._env.physics.suppress_physics_errors():
            step_env(env, visualize=False)
        env.close()

    def test_pickleable(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DMControlEnv.from_suite(domain_name, task_name)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        # Skip rendering because it causes TravisCI to run out of memory
        # Sometimes random actions lead to physics errors
        with env._env.physics.suppress_physics_errors():
            step_env(env, visualize=False)
        round_trip.close()
        env.close()

    @pytest.mark.nightly
    @pytest.mark.parametrize('domain_name, task_name',
                             dm_control.suite.ALL_TASKS)
    def test_all_pickleable(self, domain_name, task_name):
        env = DMControlEnv.from_suite(domain_name, task_name)
        round_trip = pickle.loads(pickle.dumps(env))
        assert round_trip
        # Skip rendering because it causes TravisCI to run out of memory
        # Sometimes random actions lead to physics errors
        with env._env.physics.suppress_physics_errors():
            step_env(env, visualize=False)
        round_trip.close()
        env.close()

    def test_does_not_modify_actions(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DMControlEnv.from_suite(domain_name, task_name)
        a = env.action_space.sample()
        a_copy = copy(a)
        env.reset()
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
        env = DMControlEnv.from_suite(domain_name, task_name)
        a = env.action_space.sample()
        a_copy = copy(a)
        env.reset()
        env.step(a)
        if isinstance(a, collections.Iterable):
            assert a.all() == a_copy.all()
        else:
            assert a == a_copy
        env.close()

    def test_catch_no_reset(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DMControlEnv.from_suite(domain_name, task_name)
        with pytest.raises(RuntimeError, match='reset()'):
            env.step(1)

    def test_done_resets_step_cnt(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DMControlEnv.from_suite(domain_name, task_name)
        max_episode_length = env.spec.max_episode_length

        env.reset()
        for _ in range(int(max_episode_length)):
            es = env.step(env.action_space.sample())
            if es.last:
                break
        assert env._step_cnt is None

    def test_timeout(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DMControlEnv.from_suite(domain_name, task_name)
        max_episode_length = env.spec.max_episode_length

        env.reset()
        for _ in range(int(max_episode_length)):
            es = env.step(env.action_space.sample())
        assert es.terminal

    def test_visualization(self):
        domain_name, task_name = dm_control.suite.ALL_TASKS[0]
        env = DMControlEnv.from_suite(domain_name, task_name)

        for mode in env.render_modes:
            env.render(mode)

        env.reset()
        env.visualize()
        env.step(env.action_space.sample())

        env.close()
