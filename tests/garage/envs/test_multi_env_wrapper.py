"""Tests for garage.envs.multi_env_wrapper"""
import akro
import numpy as np
import pytest

from garage.envs import GymEnv
from garage.envs.multi_env_wrapper import (MultiEnvWrapper,
                                           round_robin_strategy,
                                           uniform_random_strategy)


class TestMultiEnvWrapper:
    """Tests for garage.envs.multi_env_wrapper"""

    def _init_multi_env_wrapper(self,
                                env_names,
                                sample_strategy=uniform_random_strategy):
        """helper function to initialize multi_env_wrapper

        Args:
            env_names (list(str)): List of Environment names.
            sample_strategy (func): A sampling strategy.

        Returns:
            garage.envs.multi_env_wrapper: Multi env wrapper.
        """
        task_envs = [GymEnv(name) for name in env_names]
        return MultiEnvWrapper(task_envs, sample_strategy=sample_strategy)

    def test_tasks_from_same_env(self):
        """test init with multiple tasks from same env"""
        envs = ['CartPole-v0', 'CartPole-v0']
        mt_env = self._init_multi_env_wrapper(envs)
        assert mt_env.num_tasks == 2

    def test_tasks_from_different_envs(self):
        """test init with multiple tasks from different env"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(envs)
        assert mt_env.num_tasks == 2

    def test_raise_exception_when_different_obs_space(self):
        """test if exception is raised when using tasks with different obs space"""  # noqa: E501
        envs = ['CartPole-v0', 'Blackjack-v0']
        with pytest.raises(ValueError):
            _ = self._init_multi_env_wrapper(envs)

    def test_raise_exception_when_different_action_space(self):
        """test if exception is raised when using tasks with different action space"""  # noqa: E501
        envs = ['LunarLander-v2', 'LunarLanderContinuous-v2']
        with pytest.raises(ValueError):
            _ = self._init_multi_env_wrapper(envs)

    def test_default_active_task_is_none(self):
        """test if default active task is none"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(
            envs, sample_strategy=round_robin_strategy)
        assert mt_env._active_task_index is None

    def test_one_hot_observation_space(self):
        """test one hot representation of observation space"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(envs)
        cartpole = GymEnv('CartPole-v0')
        cartpole_lb, cartpole_ub = cartpole.observation_space.bounds
        obs_space = akro.Box(np.concatenate([cartpole_lb,
                                             np.zeros(2)]),
                             np.concatenate([cartpole_ub,
                                             np.ones(2)]))
        assert mt_env.observation_space.shape == obs_space.shape
        assert (
            mt_env.observation_space.bounds[0] == obs_space.bounds[0]).all()
        assert (
            mt_env.observation_space.bounds[1] == obs_space.bounds[1]).all()

    def test_action_space(self):
        """test action space"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(envs)
        task1 = GymEnv('CartPole-v0')
        assert mt_env.action_space.shape == task1.action_space.shape

    def test_round_robin_sample_strategy(self):
        """test round robin samping strategy"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(
            envs, sample_strategy=round_robin_strategy)
        tasks = []
        for _ in envs:
            mt_env.reset()
            es = mt_env.step(1)
            tasks.append(es.env_info['task_id'])

        assert tasks[0] == 0 and tasks[1] == 1

    def test_uniform_random_sample_strategy(self):
        """test uniform_random sampling strategy"""
        envs = ['CartPole-v0', 'CartPole-v1', 'CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(
            envs, sample_strategy=uniform_random_strategy)
        tasks = []
        for _ in envs:
            mt_env.reset()
            es = mt_env.step(1)
            tasks.append(es.env_info['task_id'])

        for task in tasks:
            assert 0 <= task < 4

    def test_task_remains_same_between_multiple_step_calls(self):
        """test if active_task remains same between multiple step calls"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(
            envs, sample_strategy=round_robin_strategy)
        mt_env.reset()
        tasks = []
        for _ in envs:
            es = mt_env.step(1)
            tasks.append(es.env_info['task_id'])

        assert tasks[0] == 0 and tasks[1] == 0

    def test_task_space(self):
        """test task space"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(envs)
        bounds = mt_env.task_space.bounds
        lb = np.zeros(2)
        ub = np.ones(2)
        assert (bounds[0] == lb).all() and (bounds[1] == ub).all()

    def test_one_hot_observation(self):
        """test if output of step function is correct"""
        envs = ['CartPole-v0', 'CartPole-v0']
        mt_env = self._init_multi_env_wrapper(
            envs, sample_strategy=round_robin_strategy)

        obs, _ = mt_env.reset()
        assert (obs[-2:] == np.array([1., 0.])).all()
        obs = mt_env.step(1).observation
        assert (obs[-2:] == np.array([1., 0.])).all()

        obs, _ = mt_env.reset()
        assert (obs[-2:] == np.array([0., 1.])).all()
        obs = mt_env.step(1).observation
        assert (obs[-2:] == np.array([0., 1.])).all()

    def test_visualization(self):
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(envs)
        mt_env.visualize()

        gym_env = GymEnv('CartPole-v0')
        assert gym_env.render_modes == mt_env.render_modes
        mode = gym_env.render_modes[0]
        assert gym_env.render(mode) == mt_env.render(mode)


@pytest.mark.mujoco
class TestMetaWorldMultiEnvWrapper:
    """Tests for garage.envs.multi_env_wrapper using Metaworld Envs"""

    def setup_class(self):
        """Init Wrapper with MT10."""
        # pylint: disable=import-outside-toplevel
        from metaworld.benchmarks import MT10
        tasks = MT10.get_train_tasks().all_task_names
        envs = []
        for task in tasks:
            envs.append(GymEnv(MT10.from_task(task)))
        self.task_names = tasks
        self.env = MultiEnvWrapper(envs,
                                   sample_strategy=round_robin_strategy,
                                   mode='vanilla',
                                   env_names=tasks)
        self.env_no_onehot = MultiEnvWrapper(
            envs, sample_strategy=round_robin_strategy, mode='del-onehot')

    def teardown_class(self):
        """Close the MTMetaWorldWrapper."""
        self.env.close()
        self.env_no_onehot.close()

    def test_num_tasks(self):
        """Assert num tasks returns 10, because MT10 is being tested."""
        assert self.env.num_tasks == 10
        assert self.env_no_onehot.num_tasks == 10

    def test_observation_space(self):
        assert self.env.observation_space.shape == (9 + self.env.num_tasks, )
        assert self.env_no_onehot.observation_space.shape == (9, )

    def test_step(self):
        """Test that env_infos includes extra infos and obs has onehot."""
        self.env.reset()
        self.env_no_onehot.reset()
        action0 = self.env.spec.action_space.sample()
        action1 = self.env_no_onehot.spec.action_space.sample()
        es = self.env.step(action0)
        obs0, info0 = es.observation, es.env_info
        es = self.env_no_onehot.step(action1)
        obs1, info1 = es.observation, es.env_info
        assert info0['task_id'] == self.env.active_task_index
        assert info1['task_id'] == self.env.active_task_index
        assert (self.env._active_task_one_hot() == obs0[9:]).all()
        assert obs0.shape[0] == obs1.shape[0] + self.env.num_tasks

    def test_reset(self):
        """Test round robin switching of environments during call to reset."""
        self.env.reset()
        active_task_id = self.env.active_task_index
        for _ in range(self.env.num_tasks):
            self.env.reset()
            action = self.env.spec.action_space.sample()
            info = self.env.step(action).env_info
            assert not info['task_id'] == active_task_id
            active_task_id = self.env.active_task_index

    def test_env_names(self):
        """Test that env_names that are passed via the env_names param are added to env_infos."""  # noqa: E501
        self.env.reset()
        self.env._active_task_index = 0
        for i in range(self.env.num_tasks):
            action = self.env.spec.action_space.sample()
            info = self.env.step(action).env_info
            assert info['task_name'] == self.task_names[i]
            self.env.reset()
