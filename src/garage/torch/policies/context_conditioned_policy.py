# pylint: disable=attribute-defined-outside-init, protected-access
"""A policy used in training meta reinforcement learning algorithms.

It is used in PEARL (Probabilistic Embeddings for Actor-Critic Reinforcement
Learning). The paper on PEARL can be found at https://arxiv.org/abs/1903.08254.
Code is adapted from https://github.com/katerakelly/oyster.
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

import garage.torch.utils as tu


class ContextConditionedPolicy(nn.Module):
    """A policy that outputs actions based on observation and latent context.

    In PEARL, policies are conditioned on current state and a latent context
    (adaptation data) variable Z. This inference network estimates the
    posterior probability of z given past transitions. It uses context
    information stored in the encoder to infer the probabilistic value of z and
    samples from a policy conditioned on z.

    Args:
        latent_dim (int): Latent context variable dimension.
        context_encoder (garage.torch.embeddings): Recurrent or
            permutation-invariant context encoder.
        policy (garage.torch.policies.Policy): Policy used to train the
            network.
        use_ib (bool): True if latent context is not deterministic; false
            otherwise
        use_next_obs (bool): True if next observation is used in context
            for distinguishing tasks; false otherwise.

    """

    def __init__(self, latent_dim, context_encoder, policy, use_ib,
                 use_next_obs):
        super().__init__()
        self._latent_dim = latent_dim
        self._context_encoder = context_encoder
        self._policy = policy
        self._use_ib = use_ib
        self._use_next_obs = use_next_obs

        # initialize buffers for z distribution and z
        # use buffers so latent context can be saved along with model weights
        # z_means and z_vars are the params for the gaussian distribution
        # over latent task belief maintained in the policy; z is a sample from
        # this distribution that the policy is conditioned on
        self.register_buffer('z', torch.zeros(1, latent_dim))
        self.register_buffer('z_means', torch.zeros(1, latent_dim))
        self.register_buffer('z_vars', torch.zeros(1, latent_dim))

        self.reset_belief()

    def reset_belief(self, num_tasks=1):
        """Reset q(z|c) to the prior and sample a new z from the prior.

        Args:
            num_tasks (int): Number of tasks.

        """
        # reset distribution over z to the prior
        mu = torch.zeros(num_tasks, self._latent_dim).to(tu.global_device())
        if self._use_ib:
            var = torch.ones(num_tasks,
                             self._latent_dim).to(tu.global_device())
        else:
            var = torch.zeros(num_tasks,
                              self._latent_dim).to(tu.global_device())
        self.z_means = mu
        self.z_vars = var
        # sample a new z from the prior
        self.sample_from_belief()
        # reset the context collected so far
        self._context = None
        # reset any hidden state in the encoder network (relevant for RNN)
        self._context_encoder.reset(num_tasks)

    def sample_from_belief(self):
        """Sample z using distributions from current means and variances."""
        if self._use_ib:
            posteriors = [
                torch.distributions.Normal(m, torch.sqrt(s)) for m, s in zip(
                    torch.unbind(self.z_means), torch.unbind(self.z_vars))
            ]
            z = [d.rsample() for d in posteriors]
            self.z = torch.stack(z)
        else:
            self.z = self.z_means

    def detach_z(self):
        """Disable backprop through z."""
        self.z = self.z.detach()

    def update_context(self, inputs):
        """Append single transition to the current context.

        Args:
            inputs (dict): Dictionary of transition information in np arrays .

        """
        o, a, r, no, _, _ = inputs
        o = torch.as_tensor(o[None, None, ...],
                            device=tu.global_device()).float()
        a = torch.as_tensor(a[None, None, ...],
                            device=tu.global_device()).float()
        r = torch.as_tensor(np.array([r])[None, None, ...],
                            device=tu.global_device()).float()
        no = torch.as_tensor(no[None, None, ...],
                             device=tu.global_device()).float()

        if self._use_next_obs:
            data = torch.cat([o, a, r, no], dim=2)
        else:
            data = torch.cat([o, a, r], dim=2)

        if self._context is None:
            self._context = data
        else:
            self._context = torch.cat([self._context, data], dim=1)

    def infer_posterior(self, context):
        """Compute q(z|c) as a function of input context and sample new z.

        Args:
            context (torch.Tensor): Context values.

        """
        params = self._context_encoder(context)
        params = params.view(context.size(0), -1,
                             self._context_encoder._output_dim)
        # with probabilistic z, predict mean and variance of q(z | c)
        if self._use_ib:
            mu = params[..., :self._latent_dim]
            sigma_squared = F.softplus(params[..., self._latent_dim:])
            z_params = [
                tu.product_of_gaussians(m, s)
                for m, s in zip(torch.unbind(mu), torch.unbind(sigma_squared))
            ]
            self.z_means = torch.stack([p[0] for p in z_params])
            self.z_vars = torch.stack([p[1] for p in z_params])
        else:
            self.z_means = torch.mean(params, dim=1)
        self.sample_from_belief()

    # pylint: disable=arguments-differ
    def forward(self, obs, context):
        """Given observations and context, get actions and probs from policy.

        Args:
            obs (torch.Tensor): Observation values.
            context (torch.Tensor): Context values.

        Returns:
            tuple:
                * torch.Tensor: Predicted action values.
                * np.ndarray: Mean of distribution.
                * np.ndarray: Log std of distribution.
                * torch.Tensor: Log likelihood of distribution.
                * torch.Tensor: Sampled values from distribution before
                    applying tanh transformation.
            torch.Tensor: z values.

        """
        self.infer_posterior(context)
        self.sample_from_belief()
        task_z = self.z

        # task, batch
        t, b, _ = obs.size()
        obs = obs.view(t * b, -1)
        task_z = [z.repeat(b, 1) for z in task_z]
        task_z = torch.cat(task_z, dim=0)

        # run policy, get log probs and new actions
        obs_z = torch.cat([obs, task_z.detach()], dim=1)
        dist = self._policy(obs_z)
        pre_tanh, actions = dist.base_dist.rsample_with_pre_tanh_value()
        log_pi = dist.base_dist.log_prob(value=actions,
                                         pre_tanh_value=pre_tanh)
        log_pi = log_pi.unsqueeze(1)
        mean = dist.mean.to('cpu').detach().numpy()
        log_std = (dist.variance**.5).log().to('cpu').detach().numpy()

        return (actions, mean, log_std, log_pi, pre_tanh), task_z

    def get_action(self, obs):
        """Sample action from the policy, conditioned on the task embedding.

        Args:
            obs (torch.Tensor): Observation values.

        Returns:
            torch.Tensor: Output action values.

        """
        z = self.z
        obs = torch.as_tensor(obs[None], device=tu.global_device()).float()
        obs_in = torch.cat([obs, z], dim=1)
        action, info = self._policy.get_action(obs_in)
        action = np.squeeze(action, axis=0)
        info['mean'] = np.squeeze(info['mean'], axis=0)
        return action, info

    def compute_kl_div(self):
        r"""Compute :math:`KL(q(z|c) \| p(z))`.

        Returns:
            float: :math:`KL(q(z|c) \| p(z))`.

        """
        prior = torch.distributions.Normal(
            torch.zeros(self._latent_dim).to(tu.global_device()),
            torch.ones(self._latent_dim).to(tu.global_device()))
        posteriors = [
            torch.distributions.Normal(mu, torch.sqrt(var)) for mu, var in zip(
                torch.unbind(self.z_means), torch.unbind(self.z_vars))
        ]
        kl_divs = [
            torch.distributions.kl.kl_divergence(post, prior)
            for post in posteriors
        ]
        kl_div_sum = torch.sum(torch.stack(kl_divs))
        return kl_div_sum

    def reset(self, dones=None):
        """Reset the environment.

        Args:
            dones (numpy.ndarray): Reset values.

        """

    def log_diagnostics(self, eval_statistics):
        """Log data about encodings to eval_statistics dictionary.

        Args:
            eval_statistics (dict): Dictionary for logging info.

        """
        z_mean = np.mean(np.abs(self.z_means[0].to('cpu').detach().numpy()))
        z_sig = np.mean(self.z_vars[0].to('cpu').detach().numpy())
        eval_statistics['ZMeanEval'] = z_mean
        eval_statistics['ZVarianceEval'] = z_sig

    @property
    def networks(self):
        """Return context_encoder and policy.

        Returns:
            list: Encoder and policy networks.

        """
        return [self._context_encoder, self._policy]

    @property
    def context(self):
        """Return context.

        Returns:
            torch.Tensor: Context values.

        """
        return self._context
