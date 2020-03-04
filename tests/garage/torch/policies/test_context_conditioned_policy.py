"""This is a script to test the ContextConditionedPolicy module."""

import akro
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F  # NOQA

from garage.envs.env_spec import EnvSpec
from garage.tf.envs import TfEnv
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import ContextConditionedPolicy
from garage.torch.policies import TanhGaussianMLPPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContextConditionedPolicy:
    """Test for ContextConditionedPolicy."""
    # yapf: disable
    @pytest.mark.parametrize('latent_dim, hidden_sizes, updates', [
        (5, (3, 3, 3), 5),
        (7, (5, 7, 5), 7),
        (10, (9, 4, 2), 10),
    ])
    # yapf: enable
    def test_module(self, latent_dim, hidden_sizes, updates):
        """Test all methods."""
        env_spec = TfEnv(DummyBoxEnv())
        latent_space = akro.Box(low=-1,
                                high=1,
                                shape=(latent_dim, ),
                                dtype=np.float32)

        # add latent space to observation space to create a new space
        augmented_obs_space = akro.Tuple(
            (env_spec.observation_space, latent_space))
        augmented_env_spec = EnvSpec(augmented_obs_space,
                                     env_spec.action_space)

        obs_dim = int(np.prod(env_spec.observation_space.shape))
        action_dim = int(np.prod(env_spec.action_space.shape))
        reward_dim = 1
        encoder_input_dim = obs_dim + action_dim + reward_dim
        encoder_output_dim = latent_dim * 2
        encoder_hidden_sizes = (3, 2, encoder_output_dim)

        context_encoder = MLPEncoder(input_dim=encoder_input_dim,
                                     output_dim=encoder_output_dim,
                                     hidden_nonlinearity=None,
                                     hidden_sizes=encoder_hidden_sizes,
                                     hidden_w_init=nn.init.ones_,
                                     output_w_init=nn.init.ones_)

        context_policy = TanhGaussianMLPPolicy(env_spec=augmented_env_spec,
                                               hidden_sizes=hidden_sizes,
                                               hidden_nonlinearity=F.relu,
                                               output_nonlinearity=None)

        module = ContextConditionedPolicy(latent_dim=latent_dim,
                                          context_encoder=context_encoder,
                                          policy=context_policy,
                                          use_ib=True,
                                          use_next_obs=False)

        module.reset()
        expected_shape = [1, latent_dim]
        module.reset_belief()
        assert torch.all(torch.eq(module.z_means, torch.zeros(expected_shape)))
        assert torch.all(torch.eq(module.z_vars, torch.ones(expected_shape)))

        module.sample_from_belief()
        assert all([a == b for a, b in zip(module.z.shape, expected_shape)])

        module.detach_z()
        assert not module.z.requires_grad

        o = np.ones(obs_dim)
        a = np.ones(action_dim)
        r = 1
        no = np.ones(obs_dim)
        dummy = np.ones(obs_dim)

        for _ in range(updates):
            module.update_context([o, a, r, no, dummy, dummy])
        assert torch.all(
            torch.eq(module.context, torch.ones(updates, encoder_input_dim)))

        context = torch.randn(1, 1, encoder_input_dim)
        module.infer_posterior(context)
        assert all([a == b for a, b in zip(module.z.shape, expected_shape)])

        t, b = 1, 2
        obs = torch.randn((t, b, obs_dim), dtype=torch.float32)
        policy_output, task_z_out = module.forward(obs, context)

        expected_shape = [b, action_dim]
        assert all(
            [a == b for a, b in zip(policy_output[0].shape, expected_shape)])
        expected_shape = [b, latent_dim]
        assert all([a == b for a, b in zip(task_z_out.shape, expected_shape)])

        obs = np.random.rand(obs_dim)
        action, _ = module.get_action(obs)
        assert len(action) == action_dim

        kl_div = module.compute_kl_div()
        assert kl_div != 0

        log_dict = {}
        module.log_diagnostics(log_dict)
        assert 'ZMeanEval' in log_dict and 'ZVarianceEval' in log_dict

        nets = module.networks
        assert nets[0] and nets[1]
