"""Vanilla Policy Gradient (REINFORCE)."""
import collections
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import log_performance
from garage.np.algos import RLAlgorithm
from garage.torch import (as_tensor, compute_advantages, filter_valids,
                          global_device, ObservationBatch, ObservationOrder)
from garage.torch._functions import (discount_cumsum, pad_packed_tensor,
                                     split_packed_tensor)
from garage.torch.optimizers import MinibatchOptimizer, SingleBatchOptimizer


class VPG(RLAlgorithm):
    """Vanilla Policy Gradient (REINFORCE).

    VPG, also known as Reinforce, trains stochastic policy in an on-policy way.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        sampler (garage.sampler.Sampler): Sampler.
        policy_optimizer (garage.torch.optimizer.Optimizer): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.Optimizer): Optimizer for
            value function.
        steps_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.

    """

    def __init__(
        self,
        env_spec,
        policy,
        value_function,
        sampler,
        policy_optimizer=None,
        vf_optimizer=None,
        steps_per_epoch=1,
        discount=0.99,
        gae_lambda=1,
        center_adv=True,
        positive_adv=False,
        policy_ent_coeff=0.0,
        use_neg_logli_entropy=True,
        use_softplus_entropy=False,
        stop_entropy_gradient=True,
        entropy_method='no_entropy',
        recurrent=None,
    ):
        self._discount = discount
        self.policy = policy
        self.max_episode_length = env_spec.max_episode_length

        self._value_function = value_function
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._use_neg_logli_entropy = use_neg_logli_entropy
        self._entropy_method = entropy_method
        self._steps_per_epoch = steps_per_epoch
        self._env_spec = env_spec

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=100)
        self._sampler = sampler
        if recurrent is None:
            recurrent = is_policy_recurrent(policy)
        self._recurrent = recurrent

        if policy_optimizer:
            self._policy_optimizer = policy_optimizer
        elif self._recurrent:
            self._policy_optimizer = EpisodeBatchOptimizer(
                torch.optim.Adam, policy)
        else:
            self._policy_optimizer = SingleBatchOptimizer(
                torch.optim.Adam, policy)
        if vf_optimizer:
            self._vf_optimizer = vf_optimizer
        elif self._recurrent:
            self._vf_optimizer = EpisodeBatchOptimizer(torch.optim.Adam,
                                                       value_function)
        else:
            self._vf_optimizer = MinibatchOptimizer(torch.optim.Adam,
                                                    value_function)

        self._old_policy = copy.deepcopy(self.policy)

    @staticmethod
    def _check_entropy_configuration(entropy_method, center_adv,
                                     stop_entropy_gradient, policy_ent_coeff):
        if entropy_method not in ('max', 'regularized', 'no_entropy'):
            raise ValueError('Invalid entropy_method')

        if entropy_method == 'max':
            if center_adv:
                raise ValueError('center_adv should be False when '
                                 'entropy_method is max')
            if not stop_entropy_gradient:
                raise ValueError('stop_gradient should be True when '
                                 'entropy_method is max')
        if entropy_method == 'no_entropy':
            if policy_ent_coeff != 0.0:
                raise ValueError('policy_ent_coeff should be zero '
                                 'when there is no entropy method')

    @property
    def discount(self):
        """Discount factor used by the algorithm.

        Returns:
            float: discount factor.
        """
        return self._discount

    def _train_once(self, eps):
        """Train the algorithm once.

        Args:
            eps (EpisodeBatch): A batch of collected paths.

        """
        # Conver to torch and compute returns, etc.
        lengths = torch.from_numpy(eps.lengths).to(global_device())
        obs = ObservationBatch(as_tensor(eps.observations),
                               order=ObservationOrder.EPISODES,
                               lengths=lengths)
        actions = torch.Tensor(eps.actions)
        rewards = torch.Tensor(eps.rewards)
        policy_entropies = self._compute_policy_entropy(obs, actions)
        if self._maximum_entropy:
            rewards += self._policy_ent_coeff * policy_entropies
        returns = torch.hstack([
            discount_cumsum(r, self._discount)
            for r in split_packed_tensor(rewards, lengths)
        ])
        with torch.no_grad():
            baselines = self._value_function(obs)
        advantages = self._compute_advantage(rewards, lengths, baselines)

        # Log before training
        with torch.no_grad():
            policy_loss_before = self._loss_function(obs, actions, rewards,
                                                     advantages, lengths)
            vf_loss_before = self._value_function.loss_function(obs, returns)

            with tabular.prefix(self.policy.name):
                tabular.record('/LossBefore', policy_loss_before.item())
                tabular.record('/KLBefore',
                               self._compute_kl_constraint(obs).item())
                tabular.record('/EntropyBefore',
                               policy_entropies.mean().item())

            with tabular.prefix(self._value_function.name):
                tabular.record('/LossBefore', vf_loss_before.item())

        # Save the current policy state and train
        self._old_policy.load_state_dict(self.policy.state_dict())
        self._train_policy(obs, actions, rewards, returns, advantages, lengths)
        self._train_value_function(obs, returns, lengths)

        # Log after training
        with torch.no_grad():

            policy_loss_after = self._loss_function(obs, actions, rewards,
                                                    advantages, lengths)
            with tabular.prefix(self.policy.name):
                tabular.record('/LossAfter', policy_loss_after.item())
                tabular.record('/dLoss',
                               (policy_loss_before - policy_loss_after).item())
                tabular.record('/KL', self._compute_kl_constraint(obs).item())
                tabular.record(
                    '/EntropyAfter',
                    self._compute_policy_entropy(obs, actions).mean().item())

            vf_loss_after = self._value_function.loss_function(obs, returns)
            with tabular.prefix(self._value_function.name):
                tabular.record('/vfLossAfter', vf_loss_after.item())
                tabular.record('/dLoss',
                               vf_loss_before.item() - vf_loss_after.item())

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Gives the algorithm the access to
                :method:`~Trainer.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for epoch in trainer.step_epochs():
            for _ in range(self._steps_per_epoch):
                trainer.step_path = self._sampler.obtain_episodes(epoch)
                self._train_once(trainer.step_path)
            last_return = np.mean(
                log_performance(epoch,
                                trainer.step_path,
                                discount=self._discount))
        return last_return

    def _train_policy(self, observations, actions, rewards, returns,
                      advantages, lengths):
        r"""Train the policy.

        Args:
            observations (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N, )`.
            lengths (torch.Tensor): Lengths of episodes.

        Returns:
            torch.Tensor: Calculated mean scalar value of policy loss (float).

        """
        data = {
            'observations': observations,
            'actions': actions,
            'rewards': rewards,
            'advantages': advantages,
        }
        if not isinstance(self._policy_optimizer, MinibatchOptimizer):
            data['lengths'] = lengths
        return self._policy_optimizer.step(data, self._loss_function)

    def _train_value_function(self, observations, returns, lengths):
        r"""Train the value function.

        Args:
            observations (torch.Tensor): Observation from the environment
                with shape :math:`(N, O*)`.
            returns (torch.Tensor): Acquired returns
                with shape :math:`(N, )`.
            lengths (torch.Tensor): Lengths of episodes.

        Returns:
            torch.Tensor: Calculated mean scalar value of value function loss
                (float).

        """
        data = {
            'observations': observations,
            'returns': returns,
            'lengths': lengths
        }
        return self._vf_optimizer.step(data,
                                       self._value_function.loss_function)

    def _compute_loss(self, obs, actions, rewards, lengths, baselines):
        r"""Compute mean value of loss.

        Note that this function is private, but used by MAML.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N, P, A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            lengths (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """
        obs_flat = torch.cat(filter_valids(obs, lengths))
        actions_flat = torch.cat(filter_valids(actions, lengths))
        rewards_flat = torch.cat(filter_valids(rewards, lengths))
        advantages_flat = self._compute_advantage(rewards, lengths, baselines)

        return self._loss_function(obs_flat, actions_flat, rewards_flat,
                                   advantages_flat, lengths)

    def _loss_function(self,
                       observations,
                       actions,
                       rewards,
                       advantages,
                       lengths=None):
        r"""Compute mean value of loss.

        Args:
            observations (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            lengths (torch.Tensor or None): Lengths of episodes, if operating
                on full episodes.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of objective.

        """
        objectives = self._compute_objective(advantages, observations, actions,
                                             rewards)

        if self._entropy_regularzied:
            policy_entropies = self._compute_policy_entropy(
                observations, actions)
            objectives += self._policy_ent_coeff * policy_entropies

        loss = -objectives.mean()
        return loss

    def _compute_advantage(self, rewards, lengths, baselines):
        r"""Compute mean value of loss.

        Args:
            rewards (torch.Tensor): Packed acquired rewards
                with shape :math:`(N \bullet [T])`.
            lengths (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Packed value function estimation of
                returns with shape :math:`(N \bullet [T])`.

        Returns:
            torch.Tensor: Calculated advantage values given rewards and
                baselines with shape :math:`(N \dot [T], )`.

        """
        padded_rewards = pad_packed_tensor(rewards, lengths)
        padded_baselines = pad_packed_tensor(baselines, lengths)
        padded_advantages = compute_advantages(self._discount,
                                               self._gae_lambda,
                                               self.max_episode_length,
                                               padded_baselines,
                                               padded_rewards)
        advantages = torch.cat(filter_valids(padded_advantages, lengths))

        if self._center_adv:
            means = advantages.mean()
            variance = advantages.var()
            advantages = (advantages - means) / (variance + 1e-8)

        if self._positive_adv:
            advantages -= advantages.min()

        return advantages

    def _compute_kl_constraint(self, obs):
        r"""Compute KL divergence.

        Compute the KL divergence between the old policy distribution and
        current policy distribution.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.

        Returns:
            torch.Tensor: Calculated mean scalar value of KL divergence
                (float).

        """
        with torch.no_grad():
            old_dist = self._old_policy(obs)[0]

        new_dist = self.policy(obs)[0]

        kl_constraint = torch.distributions.kl.kl_divergence(
            old_dist, new_dist)

        return kl_constraint.mean()

    def _compute_policy_entropy(self, obs, actions):
        r"""Compute entropy value of probability distribution.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            observations (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.

        Returns:
            torch.Tensor: Calculated entropy values given observation
                with shape :math:`(N, P)`.

        """
        with torch.set_grad_enabled(not self._stop_entropy_gradient):
            if self._use_neg_logli_entropy:
                policy_entropy = -self.policy(obs)[0].log_prob(actions)
            else:
                policy_entropy = self.policy(obs)[0].entropy()

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_objective(self, advantages, obs, actions, rewards):
        r"""Compute objective value.

        Args:
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated objective values
                with shape :math:`(N \dot [T], )`.

        """
        del rewards
        log_likelihoods = self.policy(obs)[0].log_prob(actions)

        return log_likelihoods * advantages
