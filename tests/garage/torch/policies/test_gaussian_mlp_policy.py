"""Test Gaussian MLP Policy."""
import pickle

import numpy as np
import pytest
import torch
from torch import nn

from garage.envs import GymEnv
from garage.torch.policies import GaussianMLPPolicy

# yapf: Disable
from tests.fixtures.envs.dummy import DummyBoxEnv, DummyDictEnv

# yapf: Enable


class TestGaussianMLPPolicies:
    """Class for Testing Gaussian MlP Policy."""
    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 4), (3, 5)])
    # yapf: enable
    def test_get_action(self, hidden_sizes):
        """Test get_action function."""
        env_spec = GymEnv(DummyBoxEnv())
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

        dist = policy(obs)[0]

        expected_mean = torch.full(
            (act_dim, ),
            obs_dim * (torch.Tensor(hidden_sizes).prod().item()),
            dtype=torch.float)
        expected_variance = init_std**2
        action, prob = policy.get_action(obs)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(
            torch.full((act_dim, ), expected_variance, dtype=torch.float))
        assert action.shape == (act_dim, )

    # yapf: disable
    @pytest.mark.parametrize('hidden_sizes', [
        (1, ), (2, ), (3, ), (1, 4), (3, 5)])
    # yapf: enable
    def test_get_action_np(self, hidden_sizes):
        """Test get_action function with numpy inputs."""
        env_spec = GymEnv(DummyBoxEnv())
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

        dist = policy(torch.from_numpy(obs))[0]

        expected_mean = torch.full(
            (act_dim, ),
            obs_dim * (torch.Tensor(hidden_sizes).prod().item()),
            dtype=torch.float)
        expected_variance = init_std**2
        action, prob = policy.get_action(obs)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(
            torch.full((act_dim, ), expected_variance, dtype=torch.float))
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
        env_spec = GymEnv(DummyBoxEnv())
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

        dist = policy(obs)[0]

        expected_mean = torch.full([batch_size, act_dim],
                                   obs_dim *
                                   (torch.Tensor(hidden_sizes).prod().item()),
                                   dtype=torch.float)
        expected_variance = init_std**2
        action, prob = policy.get_actions(obs)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(
            torch.full((batch_size, act_dim),
                       expected_variance,
                       dtype=torch.float))
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
        env_spec = GymEnv(DummyBoxEnv())
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

        dist = policy(torch.from_numpy(obs))[0]

        expected_mean = torch.full([batch_size, act_dim],
                                   obs_dim *
                                   (torch.Tensor(hidden_sizes).prod().item()),
                                   dtype=torch.float)
        expected_variance = init_std**2
        action, prob = policy.get_actions(obs)

        assert np.array_equal(prob['mean'], expected_mean.numpy())
        assert dist.variance.equal(
            torch.full((batch_size, act_dim),
                       expected_variance,
                       dtype=torch.float))
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
        env_spec = GymEnv(DummyBoxEnv())
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

    def test_get_action_dict_space(self):
        """Test if observations from dict obs spaces are properly flattened."""
        env = GymEnv(DummyDictEnv(obs_space_type='box', act_space_type='box'))
        policy = GaussianMLPPolicy(env_spec=env.spec,
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
        actions, _ = policy.get_actions(np.array([obs, obs]))
        for action in actions:
            assert env.action_space.shape == action.shape
