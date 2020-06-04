"""Test Gaussian MLP Policy."""
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from garage.envs import GarageEnv
from garage.torch.policies import GaussianMLPPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestGaussianMLPPolicies:
    """Class for Testing Gaussian MlP Policy."""
    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 4), (3, 5)])
    # yapf: enable
    def test_get_action(self, hidden_sizes):
        """Test get_action function."""
        env_spec = GarageEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones(obs_dim, dtype=torch.float32)
        init_std = 2.

        policy = GaussianMLPPolicy(env_spec=env_spec,
                                   hidden_sizes=hidden_sizes,
                                   init_std=init_std,
                                   hidden_nonlinearity=None,
                                   std_parameterization='exp',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)

        dist = policy(obs)

        expected_mean = torch.full(
            (act_dim, ), obs_dim * (torch.Tensor(hidden_sizes).prod().item()))
        expected_variance = init_std**2
        action, prob = policy.get_action(obs)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(torch.full((act_dim, ), expected_variance))
        assert action.shape == (act_dim, )

    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 4), (3, 5)])
    # yapf: enable
    def test_get_action_np(self, hidden_sizes):
        """Test get_action function with numpy inputs."""
        env_spec = GarageEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = np.ones(obs_dim, dtype=np.float32)
        init_std = 2.

        policy = GaussianMLPPolicy(env_spec=env_spec,
                                   hidden_sizes=hidden_sizes,
                                   init_std=init_std,
                                   hidden_nonlinearity=None,
                                   std_parameterization='exp',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)

        dist = policy(torch.from_numpy(obs))

        expected_mean = torch.full(
            (act_dim, ), obs_dim * (torch.Tensor(hidden_sizes).prod().item()))
        expected_variance = init_std**2
        action, prob = policy.get_action(obs)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(torch.full((act_dim, ), expected_variance))
        assert action.shape == (act_dim, )

    # yapf: disable
    @pytest.mark.parametrize('batch_size, hidden_sizes', [
        (1, (1, )),
        (5, (3, )),
        (8, (4, )),
        (15, (1, 2)),
        (30, (3, 4, 10)),
    ])
    # yapf: enable
    def test_get_actions(self, batch_size, hidden_sizes):
        """Test get_actions function."""
        env_spec = GarageEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)
        init_std = 2.

        policy = GaussianMLPPolicy(env_spec=env_spec,
                                   hidden_sizes=hidden_sizes,
                                   init_std=init_std,
                                   hidden_nonlinearity=None,
                                   std_parameterization='exp',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)

        dist = policy(obs)

        expected_mean = torch.full([batch_size, act_dim],
                                   obs_dim *
                                   (torch.Tensor(hidden_sizes).prod().item()))
        expected_variance = init_std**2
        action, prob = policy.get_actions(obs)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(
            torch.full((batch_size, act_dim), expected_variance))
        assert action.shape == (batch_size, act_dim)

    # yapf: disable
    @pytest.mark.parametrize('batch_size, hidden_sizes', [
        (1, (1, )),
        (5, (3, )),
        (8, (4, )),
        (15, (1, 2)),
        (30, (3, 4, 10)),
    ])
    # yapf: enable
    def test_get_actions_np(self, batch_size, hidden_sizes):
        """Test get_actions function with numpy inputs."""
        env_spec = GarageEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = np.ones((batch_size, obs_dim), dtype=np.float32)
        init_std = 2.

        policy = GaussianMLPPolicy(env_spec=env_spec,
                                   hidden_sizes=hidden_sizes,
                                   init_std=init_std,
                                   hidden_nonlinearity=None,
                                   std_parameterization='exp',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)

        dist = policy(torch.from_numpy(obs))

        expected_mean = torch.full([batch_size, act_dim],
                                   obs_dim *
                                   (torch.Tensor(hidden_sizes).prod().item()))
        expected_variance = init_std**2
        action, prob = policy.get_actions(obs)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(
            torch.full((batch_size, act_dim), expected_variance))
        assert action.shape == (batch_size, act_dim)

    # yapf: disable
    @pytest.mark.parametrize('batch_size, hidden_sizes', [
        (1, (1, )),
        (6, (3, )),
        (11, (6, )),
        (25, (3, 5)),
        (34, (2, 10, 11)),
    ])
    # yapf: enable
    def test_is_pickleable(self, batch_size, hidden_sizes):
        """Test if policy is pickleable."""
        env_spec = GarageEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)
        init_std = 2.

        policy = GaussianMLPPolicy(env_spec=env_spec,
                                   hidden_sizes=hidden_sizes,
                                   init_std=init_std,
                                   hidden_nonlinearity=None,
                                   std_parameterization='exp',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)

        output1_action, output1_prob = policy.get_actions(obs)

        p = pickle.dumps(policy)
        policy_pickled = pickle.loads(p)
        output2_action, output2_prob = policy_pickled.get_actions(obs)

        assert np.array_equal(output1_prob['mean'], output2_prob['mean'])
        assert output1_action.shape == output2_action.shape

    def test_entropy(self):
        """Test get_entropy method of the policy."""
        env_spec = GarageEnv(DummyBoxEnv())
        init_std = 1.
        obs = torch.Tensor([0, 0, 0, 0]).float()
        policy = GaussianMLPPolicy(env_spec=env_spec,
                                   hidden_sizes=(1, ),
                                   init_std=init_std,
                                   hidden_nonlinearity=None,
                                   std_parameterization='exp',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)
        dist = policy(obs)
        assert torch.allclose(dist.entropy(), policy.entropy(obs))

    def test_log_prob(self):
        """Test log_prob method of the policy."""
        env_spec = GarageEnv(DummyBoxEnv())
        init_std = 1.
        obs = torch.Tensor([0, 0, 0, 0]).float()
        action = torch.Tensor([0, 0]).float()
        policy = GaussianMLPPolicy(env_spec=env_spec,
                                   hidden_sizes=(1, ),
                                   init_std=init_std,
                                   hidden_nonlinearity=None,
                                   std_parameterization='exp',
                                   hidden_w_init=nn.init.ones_,
                                   output_w_init=nn.init.ones_)
        dist = policy(obs)
        assert torch.allclose(dist.log_prob(action),
                              policy.log_likelihood(obs, action))
