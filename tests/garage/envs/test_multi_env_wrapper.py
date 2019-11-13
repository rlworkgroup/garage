"""Tests for garage.envs.multi_env_wrapper"""
import akro
import numpy as np
import pytest

from garage.envs.multi_env_wrapper import (MultiEnvWrapper,
                                           round_robin_strategy,
                                           uniform_random_strategy)
from garage.tf.envs import TfEnv


class TestMultiEnvWrapper:
    """Tests for garage.envs.multi_env_wrapper"""

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
        assert mt_env.active_task_index is None

    def test_one_hot_observation_space(self):
        """test one hot representation of observation space"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(envs)
        cartpole = TfEnv(env_name='CartPole-v0')
        cartpole_lb, cartpole_ub = cartpole.observation_space.bounds
        obs_space = akro.Box(np.concatenate([np.zeros(2), cartpole_lb]),
                             np.concatenate([np.ones(2), cartpole_ub]))
        assert mt_env.observation_space.shape == obs_space.shape
        assert (
            mt_env.observation_space.bounds[0] == obs_space.bounds[0]).all()
        assert (
            mt_env.observation_space.bounds[1] == obs_space.bounds[1]).all()

    def test_action_space(self):
        """test action space"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(envs)
        task1 = TfEnv(env_name='CartPole-v0')
        assert mt_env.action_space.shape == task1.action_space.shape

    def test_round_robin_sample_strategy(self):
        """test round robin samping strategy"""
        envs = ['CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(
            envs, sample_strategy=round_robin_strategy)
        tasks = []
        for _ in envs:
            mt_env.reset()
            _, _, _, info = mt_env.step(1)
            tasks.append(info['task_id'])

        assert tasks[0] == 0 and tasks[1] == 1

    def test_uniform_random_sample_strategy(self):
        """test uniform_random sampling strategy"""
        envs = ['CartPole-v0', 'CartPole-v1', 'CartPole-v0', 'CartPole-v1']
        mt_env = self._init_multi_env_wrapper(
            envs, sample_strategy=uniform_random_strategy)
        tasks = []
        for _ in envs:
            mt_env.reset()
            _, _, _, info = mt_env.step(1)
            tasks.append(info['task_id'])

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
            _, _, _, info = mt_env.step(1)
            tasks.append(info['task_id'])

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

        obs = mt_env.reset()
        assert (obs[:2] == np.array([1., 0.])).all()
        obs = mt_env.step(1)[0]
        assert (obs[:2] == np.array([1., 0.])).all()

        obs = mt_env.reset()
        assert (obs[:2] == np.array([0., 1.])).all()
        obs = mt_env.step(1)[0]
        assert (obs[:2] == np.array([0., 1.])).all()

    def _init_multi_env_wrapper(self,
                                env_names,
                                sample_strategy=uniform_random_strategy):
        """helper function to initialize multi_env_wrapper

        Args:
            env_names (list(str)): List of gym.Env names.
            sample_strategy (func): A sampling strategy.

        Returns:
            garage.envs.multi_env_wrapper: Multi env wrapper.
        """
        task_envs = [TfEnv(env_name=name) for name in env_names]
        return MultiEnvWrapper(task_envs, sample_strategy=sample_strategy)
