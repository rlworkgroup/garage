import numpy as np
import pickle
import pytest
import torch
import torch.nn as nn
from torch.nn import functional as F  # NOQA

from garage.tf.envs import TfEnv
from garage.torch.algos import PEARLInferenceNetwork
from garage.torch.algos import RecurrentEncoder
from garage.torch.policies import GaussianMLPPolicy
from tests.fixtures.envs.dummy import DummyBoxEnv

class TestPEARLInferenceNetwork:
    # yapf: disable
    @pytest.mark.parametrize('reward_dim, latent_dim, hidden_sizes, \
        num_update', [
        (1, 5, (3, 3, 3), 5),
        (2, 5, (5, 7, 5), 7),
        (3, 5, (9, 4, 2), 10),
    ])
    # yapf: enable
    def test_module(self, reward_dim, latent_dim, hidden_sizes, num_update):
        env_spec = TfEnv(DummyBoxEnv())
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
        #context_encoder.reset(num_tasks=encoder_input_dim)

        # policy needs to be able to accept obs_dim + latent_dim as input dim
        policy = GaussianMLPPolicy(env_spec=env_spec,
                                    hidden_sizes=hidden_sizes,
                                    hidden_nonlinearity=F.relu,
                                    output_nonlinearity=None)
      
        module = PEARLInferenceNetwork(latent_dim=latent_dim,
                                context_encoder=context_encoder,
                                policy=policy,
                                recurrent=False,
                                use_information_bottleneck=True, 
                                use_next_obs_in_context=False)
        
        expected_shape = [1, latent_dim]
        module.clear_z()
        assert torch.all(torch.eq(module.z_means, torch.zeros(expected_shape)))
        assert torch.all(torch.eq(module.z_vars, torch.ones(expected_shape)))

        module.sample_z()
        assert torch.all(torch.eq(torch.tensor(module.z.shape), 
            torch.tensor(expected_shape)))
        
        context_dict = {}
        context_dict['observation'] = np.ones(obs_dim)
        context_dict['action'] = np.ones(action_dim)
        context_dict['reward'] = np.ones(reward_dim)
        context_dict['next_observation'] = np.ones(obs_dim)

        for _ in range(num_update):
            module.update_context(context_dict)
        assert torch.all(torch.eq(module.context, torch.ones(
            num_update, encoder_input_dim)))
        
        context = torch.randn(1, 1, encoder_input_dim)
        module.infer_posterior(context)
        assert torch.all(torch.eq(torch.tensor(module.z.shape), 
            torch.tensor(expected_shape)))

        # current implementation of policy does not work because of fixed input_dim
        """
        expected_shape = [1, action_dim]
        t, b = 1, 2
        obs = torch.randn((t, b, obs_dim), dtype=torch.float32)
        policy_output, task_z_out = module.forward(obs, context)

        obs = torch.randn(obs_dim)
        action = module.get_action(obs)
        """
            

