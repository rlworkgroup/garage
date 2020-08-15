"""Tests for tanh gaussian mlp policy."""
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from garage.envs import GymEnv
from garage.torch.policies import TanhGaussianMLPPolicy

# yapf: Disable
from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDictEnv

# yapf: Enable


class TestTanhGaussianMLPPolicy:
    """Tests for TanhGaussianMLPPolicy."""
    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 4), (3, 5)])
    # yapf: enable
    def test_get_action(self, hidden_sizes):
        """Test Tanh Gaussian Policy get action function."""
        env_spec = GymEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones(obs_dim, dtype=torch.float32).unsqueeze(0)
        init_std = 2.

        policy = TanhGaussianMLPPolicy(env_spec=env_spec,
                                       hidden_sizes=hidden_sizes,
                                       init_std=init_std,
                                       hidden_nonlinearity=None,
                                       std_parameterization='exp',
                                       hidden_w_init=nn.init.ones_,
                                       output_w_init=nn.init.ones_)
        expected_mean = torch.full((act_dim, ), 1.0, dtype=torch.float)
        action, prob = policy.get_action(obs)
        assert np.allclose(prob['mean'], expected_mean.numpy(), rtol=1e-3)
        assert action.squeeze(0).shape == (act_dim, )

    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 4), (3, 5)])
    # yapf: enable
    def test_get_action_np(self, hidden_sizes):
        """Test Policy get action function with numpy inputs."""
        env_spec = GymEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = np.ones((obs_dim, ), dtype=np.float32)
        init_std = 2.

        policy = TanhGaussianMLPPolicy(env_spec=env_spec,
                                       hidden_sizes=hidden_sizes,
                                       init_std=init_std,
                                       hidden_nonlinearity=None,
                                       std_parameterization='exp',
                                       hidden_w_init=nn.init.ones_,
                                       output_w_init=nn.init.ones_)
        expected_mean = torch.full((act_dim, ), 1.0, dtype=torch.float)
        action, prob = policy.get_action(obs)
        assert np.allclose(prob['mean'], expected_mean.numpy(), rtol=1e-3)
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
        """Test Tanh Gaussian Policy get actions function."""
        env_spec = GymEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)
        init_std = 2.

        policy = TanhGaussianMLPPolicy(env_spec=env_spec,
                                       hidden_sizes=hidden_sizes,
                                       init_std=init_std,
                                       hidden_nonlinearity=None,
                                       std_parameterization='exp',
                                       hidden_w_init=nn.init.ones_,
                                       output_w_init=nn.init.ones_)

        expected_mean = torch.full([batch_size, act_dim],
                                   1.0,
                                   dtype=torch.float)
        action, prob = policy.get_actions(obs)
        assert np.allclose(prob['mean'], expected_mean.numpy(), rtol=1e-3)
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
        """Test get actions with np.ndarray inputs."""
        env_spec = GymEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        act_dim = env_spec.action_space.flat_dim
        obs = np.ones((batch_size, obs_dim), dtype=np.float32)
        init_std = 2.

        policy = TanhGaussianMLPPolicy(env_spec=env_spec,
                                       hidden_sizes=hidden_sizes,
                                       init_std=init_std,
                                       hidden_nonlinearity=None,
                                       std_parameterization='exp',
                                       hidden_w_init=nn.init.ones_,
                                       output_w_init=nn.init.ones_)

        expected_mean = torch.full([batch_size, act_dim],
                                   1.0,
                                   dtype=torch.float)
        action, prob = policy.get_actions(obs)
        assert np.allclose(prob['mean'], expected_mean.numpy(), rtol=1e-3)
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
        """Test if policy is unchanged after pickling."""
        env_spec = GymEnv(DummyBoxEnv())
        obs_dim = env_spec.observation_space.flat_dim
        obs = torch.ones([batch_size, obs_dim], dtype=torch.float32)
        init_std = 2.

        policy = TanhGaussianMLPPolicy(env_spec=env_spec,
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
        assert np.allclose(output2_prob['mean'],
                           output1_prob['mean'],
                           rtol=1e-3)
        assert output1_action.shape == output2_action.shape

    def test_to(self):
        """Test Tanh Gaussian Policy can be moved to cpu."""
        env_spec = GymEnv(DummyBoxEnv())
        init_std = 2.

        policy = TanhGaussianMLPPolicy(env_spec=env_spec,
                                       hidden_sizes=(1, ),
                                       init_std=init_std,
                                       hidden_nonlinearity=None,
                                       std_parameterization='exp',
                                       hidden_w_init=nn.init.ones_,
                                       output_w_init=nn.init.ones_)
        if torch.cuda.is_available():
            policy.to(torch.device('cuda:0'))
            assert str(next(policy.parameters()).device) == 'cuda:0'
        else:
            policy.to(None)
            assert str(next(policy.parameters()).device) == 'cpu'

    def test_get_action_dict_space(self):
        """Test if observations from dict obs spaces are properly flattened."""
        env = GymEnv(DummyDictEnv(obs_space_type='box', act_space_type='box'))
        policy = TanhGaussianMLPPolicy(env_spec=env.spec,
                                       hidden_nonlinearity=None,
                                       hidden_sizes=(1, ),
                                       hidden_w_init=nn.init.ones_,
                                       output_w_init=nn.init.ones_)
        obs = env.reset()[0]

        action, _ = policy.get_action(obs)
        assert env.action_space.shape == action.shape

        actions, _ = policy.get_actions(np.array([obs, obs]))
        for action in actions:
            assert env.action_space.shape == action.shape
