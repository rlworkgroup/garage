"""This is a script to test the ContextConditionedPolicy module."""

import akro
import numpy as np
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

    def setup_method(self):
        """Setup for all test methods."""
        self.latent_dim = 5
        env_spec = TfEnv(DummyBoxEnv())
        latent_space = akro.Box(low=-1,
                                high=1,
                                shape=(self.latent_dim, ),
                                dtype=np.float32)

        # add latent space to observation space to create a new space
        augmented_obs_space = akro.Tuple(
            (env_spec.observation_space, latent_space))
        augmented_env_spec = EnvSpec(augmented_obs_space,
                                     env_spec.action_space)

        self.obs_dim = int(np.prod(env_spec.observation_space.shape))
        self.action_dim = int(np.prod(env_spec.action_space.shape))
        reward_dim = 1
        self.encoder_input_dim = self.obs_dim + self.action_dim + reward_dim
        encoder_output_dim = self.latent_dim * 2
        encoder_hidden_sizes = (3, 2, encoder_output_dim)

        context_encoder = MLPEncoder(input_dim=self.encoder_input_dim,
                                     output_dim=encoder_output_dim,
                                     hidden_nonlinearity=None,
                                     hidden_sizes=encoder_hidden_sizes,
                                     hidden_w_init=nn.init.ones_,
                                     output_w_init=nn.init.ones_)

        context_policy = TanhGaussianMLPPolicy(env_spec=augmented_env_spec,
                                               hidden_sizes=(3, 5, 7),
                                               hidden_nonlinearity=F.relu,
                                               output_nonlinearity=None)

        self.module = ContextConditionedPolicy(latent_dim=self.latent_dim,
                                               context_encoder=context_encoder,
                                               policy=context_policy,
                                               use_ib=True,
                                               use_next_obs=False)

    def test_reset(self):
        """Test reset."""
        self.module.reset()

        expected_shape = [1, self.latent_dim]
        self.module.reset_belief()
        assert torch.all(
            torch.eq(self.module.z_means, torch.zeros(expected_shape)))
        assert torch.all(
            torch.eq(self.module.z_vars, torch.ones(expected_shape)))

    def test_sample_from_belief(self):
        """Test sample_from_belief."""
        self.module.sample_from_belief()
        expected_shape = [1, self.latent_dim]
        assert all(
            [a == b for a, b in zip(self.module.z.shape, expected_shape)])

    def test_detach_z(self):
        """Test detach_z."""
        self.module.detach_z()
        assert not self.module.z.requires_grad

    def test_update_context(self):
        """Test update_context."""
        o = np.ones(self.obs_dim)
        a = np.ones(self.action_dim)
        r = 1
        no = np.ones(self.obs_dim)
        dummy = np.ones(self.obs_dim)

        updates = 10
        for _ in range(updates):
            self.module.update_context([o, a, r, no, dummy, dummy])
        assert torch.all(
            torch.eq(self.module.context,
                     torch.ones(updates, self.encoder_input_dim)))

    def test_infer_posterior(self):
        """Test infer_posterior."""
        context = torch.randn(1, 1, self.encoder_input_dim)
        self.module.infer_posterior(context)
        expected_shape = [1, self.latent_dim]
        assert all(
            [a == b for a, b in zip(self.module.z.shape, expected_shape)])

    def test_forward(self):
        """Test forward."""
        t, b = 1, 2
        obs = torch.randn((t, b, self.obs_dim), dtype=torch.float32)
        context = torch.randn(1, 1, self.encoder_input_dim)
        policy_output, task_z_out = self.module.forward(obs, context)

        expected_shape = [b, self.action_dim]
        assert all(
            [a == b for a, b in zip(policy_output[0].shape, expected_shape)])
        expected_shape = [b, self.latent_dim]
        assert all([a == b for a, b in zip(task_z_out.shape, expected_shape)])

    def test_get_action(self):
        """Test get_action."""
        obs = np.random.rand(self.obs_dim)
        action, _ = self.module.get_action(obs)
        assert len(action) == self.action_dim

    def test_compute_kl_div(self):
        """Test compute_kl_div."""
        self.module.sample_from_belief()
        context = torch.randn(1, 1, self.encoder_input_dim)
        self.module.infer_posterior(context)
        kl_div = self.module.compute_kl_div()
        assert kl_div != 0

    def test_log_diagnostics(self):
        """Test log_diagnostics."""
        log_dict = {}
        self.module.log_diagnostics(log_dict)
        assert 'ZMeanEval' in log_dict and 'ZVarianceEval' in log_dict

    def test_networks(self):
        """Test networks."""
        nets = self.module.networks
        assert nets[0] and nets[1]
