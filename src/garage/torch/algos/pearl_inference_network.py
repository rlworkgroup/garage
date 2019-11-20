""" An inference network that learns a latent probabilistic context variable Z
used in PEARL (Probabilistic Embeddings for Actor-Critic Reinforcement 
Learning).

The paper on PEARL can be found here https://arxiv.org/abs/1903.08254.
Code is adapted from https://github.com/katerakelly/oyster.
"""
import numpy as np

import torch
from torch import nn as nn
import torch.nn.functional as F

from garage.torch.utils import np_to_torch, torch_to_np


class PEARLInferenceNetwork(nn.Module):
    """
    In PEARL, policies are conditioned on current state and a latent context
    variable Z. This inference network estimates the posterior probability of 
    z given past transitions. It uses context information stored in the 
    encoder to infer the probabilistic value of z and samples from a policy 
    conditioned on z.

    Args:
        latent_dim (int): Latent context variable dimension.
        context_encoder (garage.torch.algo.RecurrentEncoder): Recurrent 
            context encoder.
        policy (garage.torch.policies): Policy used to train the network.
        recurrent (bool): True if encoder is recurrent; false if it is 
            permutation-invariant.
        use_information_bottlebeck (bool): True if latent context is not 
            deterministic.
        use_next_obs_in_context (bool): True if next observation is used in
            distinguishing tasks.

    """
    def __init__(self,
                 latent_dim,
                 context_encoder,
                 policy,
                 **kwargs
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_encoder = context_encoder
        self.policy = policy

        self.recurrent = kwargs['recurrent']
        self.use_ib = kwargs['use_information_bottleneck']
        self.use_next_obs_in_context = kwargs['use_next_obs_in_context']

        # initialize buffers for z dist and z
        # use buffers so latent context can be saved along with model weights
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.clear_z()

    def clear_z(self, num_tasks=1):
        """
        reset q(z|c) to the prior
        sample a new z from the prior
        """
        # reset distribution over z to the prior
        mu = torch.zeros(num_tasks, self.latent_dim)
        if self.use_ib:
            var = torch.ones(num_tasks, self.latent_dim)
        else:
            var = torch.zeros(num_tasks, self.latent_dim)
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_z()
        # reset the context collected so far
        self.context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self.context_encoder.reset(num_tasks)

    def sample_z(self):
        """sample z using distributions from current means and variances"""
        if self.use_ib:
            posteriors = [torch.distributions.Normal(m, torch.sqrt(s)) \
                for m, s in zip(torch.unbind(self.z_means), \
                torch.unbind(self.z_vars))]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def detach_z(self):
        """disable backprop through z"""
        self.z = self.z.detach()
        if self.recurrent:
            self.context_encoder.hidden = self.context_encoder.hidden.detach()

    def update_context(self, inputs):
        ''' append single transition to the current context '''
        # inputs = dict of np arrays
        transitions = np_to_torch(inputs)
        o = transitions['observation']
        r = transitions['reward']
        a = transitions['action']
        no = transitions['next_observation']
        o = torch.unsqueeze(torch.unsqueeze(o, 0), 0)
        r = torch.unsqueeze(torch.unsqueeze(r, 0), 0)
        a = torch.unsqueeze(torch.unsqueeze(a, 0), 0)
        no = torch.unsqueeze(torch.unsqueeze(no, 0), 0)

        if self.use_next_obs_in_context:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)

        if self.context is None:
            self.context = data
        else:
            self.context = torch.cat([self.context, data], dim=1)

    def infer_posterior(self, context):
        """compute q(z|c) as a function of input context and sample new z"""
        params = self.context_encoder(context)
        params = params.view(context.size(0), -1, 
            self.context_encoder._output_dim)
        # given context, compute mean and variance of q(z|c)
        if self.use_ib:
            mu = params[..., :self.latent_dim]
            sigma_squared = F.softplus(params[..., self.latent_dim:])
            z_params = []
            # compute mu, sigma of product of gaussians
            for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared)):
                s = torch.clamp(s, min=1e-7)
                s = 1. / torch.sum(torch.reciprocal(s), dim=0)
                m = s * torch.sum(m / s, dim=0)
                z_params.append((m, s))
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_z()

    def forward(self, obs, context):
        """given observations and context, get actions and probs from policy"""
        self.infer_posterior(context)
        self.sample_z()
        task_z = self.z

        # task, batch
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        in_ = torch.cat([obs, task_z.detach()], dim=1)
        policy_outputs = self.policy(in_)

        return policy_outputs, task_z

    def get_action(self, obs):
        """ sample action from the policy, conditioned on the task embedding """
        z = self.z
        obs = torch.unsqueeze(obs, 0)
        in_ = torch.cat([obs, z], dim=1)
        return self.policy.get_action(in_)

    def compute_kl_div(self):
        """compute KL( q(z|c) || r(z) )"""
        prior = torch.distributions.Normal(torch.zeros(self.latent_dim), 
            torch.ones(self.latent_dim))
        posteriors = [torch.distributions.Normal(mu, torch.sqrt(var)) \
            for mu, var in zip(torch.unbind(self.z_means),
                               torch.unbind(self.z_vars))]
        kl_divs = [torch.distributions.kl.kl_divergence(post, prior) 
            for post in posteriors]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    @property
    def networks(self):
        return [self.context_encoder, self.policy]