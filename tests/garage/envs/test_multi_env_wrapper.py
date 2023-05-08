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
        return MultiEnvWrapper(task_envs,
                               mode='add-onehot',
                               sample_strategy=sample_strategy)

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

    def test_one_hot_episode_info(self):
        """test if episode info is correct"""
        envs = ['CartPole-v0', 'CartPole-v0']
        mt_env = self._init_multi_env_wrapper(
            envs, sample_strategy=round_robin_strategy)

        _, episode_info = mt_env.reset()
        assert (episode_info['task_one_hot'] == np.array([1., 0.])).all()

    def test_visualization(self):
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(envs)
        mt_env.visualize()

        gym_env = GymEnv('CartPole-v0')
        assert gym_env.render_modes == mt_env.render_modes
        mode = gym_env.render_modes[0]
        assert gym_env.render(mode) == mt_env.render(mode)
