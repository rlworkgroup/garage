# pylint: disable=missing-docstring,pointless-string-statement,protected-access
"""This is a script to test the ContextConditionedPolicy module."""

import akro
import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F  # NOQA

from garage.envs.env_spec import EnvSpec
from garage.tf.envs import TfEnv
from garage.torch.embeddings import RecurrentEncoder
from garage.torch.policies import ContextConditionedPolicy
from garage.torch.policies import GaussianMLPPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestContextConditionedPolicy:
    """Test for ContextConditionedPolicy."""
    # yapf: disable
    @pytest.mark.parametrize('reward_dim, latent_dim, hidden_sizes, updates', [
        (1, 5, (3, 3, 3), 5),
        (2, 5, (5, 7, 5), 7),
        (3, 5, (9, 4, 2), 10),
    ])
    # yapf: enable
    def test_module(self, reward_dim, latent_dim, hidden_sizes, updates):
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
        encoder_input_dim = obs_dim + action_dim + reward_dim
        encoder_output_dim = latent_dim * 2
        encoder_hidden_sizes = (3, 2, encoder_output_dim)

        context_encoder = RecurrentEncoder(input_dim=encoder_input_dim,
                                           output_dim=encoder_output_dim,
                                           hidden_nonlinearity=None,
                                           hidden_sizes=encoder_hidden_sizes,
                                           hidden_w_init=nn.init.ones_,
                                           output_w_init=nn.init.ones_)

        # policy needs to be able to accept obs_dim + latent_dim as input dim
        policy = GaussianMLPPolicy(env_spec=augmented_env_spec,
                                   hidden_sizes=hidden_sizes,
                                   hidden_nonlinearity=F.relu,
                                   output_nonlinearity=None)

        module = ContextConditionedPolicy(latent_dim=latent_dim,
                                          context_encoder=context_encoder,
                                          policy=policy,
                                          use_ib=True,
                                          use_next_obs=False)

        expected_shape = [1, latent_dim]
        module.reset_belief()
        assert torch.all(torch.eq(module.z_means, torch.zeros(expected_shape)))
        assert torch.all(torch.eq(module.z_vars, torch.ones(expected_shape)))

        module.sample_from_belief()
        assert all([a == b for a, b in zip(module.z.shape, expected_shape)])

        context_dict = {}
        context_dict['observation'] = np.ones(obs_dim)
        context_dict['action'] = np.ones(action_dim)
        context_dict['reward'] = np.ones(reward_dim)
        context_dict['next_observation'] = np.ones(obs_dim)

        for _ in range(updates):
            module.update_context(context_dict)
        assert torch.all(
            torch.eq(module._context, torch.ones(updates, encoder_input_dim)))

        context = torch.randn(1, 1, encoder_input_dim)
        module.infer_posterior(context)
        assert all([a == b for a, b in zip(module.z.shape, expected_shape)])

        # t, b = 1, 2
        # obs = torch.randn((t, b, obs_dim), dtype=torch.float32)
        # policy_output, task_z_out = module.forward(obs, context)

        obs = torch.randn(obs_dim)
        action = module.get_action(obs)
        assert len(action) == action_dim
