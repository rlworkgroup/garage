"""Advantage Actor-Critic (A2C)."""
import collections
import copy

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage import EpisodeBatch, log_performance
from garage._dtypes import StepType
from garage.misc import tensor_utils as tu
from garage.np.algos.rl_algorithm import RLAlgorithm
from garage.sampler import RaySampler
from garage.torch import compute_advantages, filter_valids, pad_to_last
from garage.torch.algos import VPG
from garage.torch.optimizers import OptimizerWrapper


class A2C(VPG):
    """A2C paper: https://arxiv.org/abs/1602.01783.

    A2C is the synchronous variant of the A3C algorithm described
    in the paper. The primary difference between A2C and A3C is that
    the former uses one policy and baseline network and
    synchronously updates these networks after all workers complete
    the sampling step.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.torch.value_functions.ValueFunction): The value
            function.
        policy_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer
            for policy.
        vf_optimizer (garage.torch.optimizer.OptimizerWrapper): Optimizer for
            value function.
        max_episode_length (int): Maximum length of a single rollout.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
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
        policy_optimizer=None,
        vf_optimizer=None,
        max_episode_length=500,
        num_train_per_epoch=1,
        discount=0.99,
        center_adv=True,
        positive_adv=False,
        policy_ent_coeff=0.01,
        use_softplus_entropy=False,
        stop_entropy_gradient=False,
        entropy_method='regularized',
    ):

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         value_function=value_function,
                         max_episode_length=max_episode_length,
                         discount=discount,
                         gae_lambda=1,
                         center_adv=center_adv,
                         num_train_per_epoch=num_train_per_epoch,
                         positive_adv=positive_adv,
                         vf_optimizer=vf_optimizer,
                         policy_optimizer=policy_optimizer,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method)

    def train_once(self, itr, paths):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Calculated mean value of undiscounted returns.

        """
        obs, actions, rewards, returns, valids, baselines = \
            self.process_samples(paths)

        if self._maximum_entropy:
            policy_entropies = self._compute_policy_entropy(obs)
            rewards += self._policy_ent_coeff * policy_entropies

        obs_flat = torch.cat(filter_valids(obs, valids))
        actions_flat = torch.cat(filter_valids(actions, valids))
        rewards_flat = torch.cat(filter_valids(rewards, valids))
        returns_flat = torch.cat(filter_valids(returns, valids))

        advs_flat = self._compute_bootstrapped_advantage(
                returns, valids, baselines)

        with torch.no_grad():
            policy_loss_before = self._compute_loss_with_adv(
                obs_flat, actions_flat, advs_flat)
            vf_loss_before = self._value_function.compute_loss(
                obs_flat, returns_flat)

        self._train_policy_and_vf(obs_flat, actions_flat, rewards_flat, returns_flat,
                    advs_flat)

        with torch.no_grad():
            policy_loss_after = self._compute_loss_with_adv(
                obs_flat, actions_flat, advs_flat)
            vf_loss_after = self._value_function.compute_loss(
                obs_flat, returns_flat)
            policy_entropy = self._compute_policy_entropy(obs)

        with tabular.prefix(self.policy.name):
            tabular.record('/LossBefore', policy_loss_before.item())
            tabular.record('/LossAfter', policy_loss_after.item())
            tabular.record('/dLoss',
                           (policy_loss_before - policy_loss_after).item())
            tabular.record('/Entropy', policy_entropy.mean().item())

        with tabular.prefix(self._value_function.name):
            tabular.record('/LossBefore', vf_loss_before.item())
            tabular.record('/LossAfter', vf_loss_after.item())
            tabular.record('/dLoss',
                           vf_loss_before.item() - vf_loss_after.item())

        self._old_policy.load_state_dict(self.policy.state_dict())

        undiscounted_returns = log_performance(itr,
                                               EpisodeBatch.from_list(
                                                   self._env_spec, paths),
                                               discount=self.discount)
        self._episode_reward_mean.extend(undiscounted_returns)
        moving_avg = np.mean(self._episode_reward_mean)
        tabular.record('Extras/100EpisodeRewardMean', moving_avg)
        return np.mean(undiscounted_returns)

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): Gives the algorithm the access to
                :method:`~LocalRunner.step_epochs()`, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            for _ in range(self._n_samples):
                runner.step_path = runner.obtain_samples(runner.step_itr)
                last_return = self.train_once(runner.step_itr,
                                              runner.step_path)
                runner.step_itr += 1

        return last_return

    def _train_policy_and_vf(self, obs, actions, rewards, returns, advs):
        r"""Train the policy and value function with minibatch.

        Args:
            obs (torch.Tensor): Observation from the environment with shape
                :math:`(N, O*)`.
            actions (torch.Tensor): Actions fed to the environment with shape
                :math:`(N, A*)`.
            rewards (torch.Tensor): Acquired rewards with shape :math:`(N, )`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.
            advs (torch.Tensor): Advantage value at each step with shape
                :math:`(N, )`.

        """
        for dataset in self._policy_optimizer.get_minibatch(
                obs, actions, advs):
            self._policy_optimizer.zero_grad()
            loss = self._compute_loss_with_adv(*dataset)
            loss.backward()
            self._policy_optimizer.step()

        for dataset in self._vf_optimizer.get_minibatch(obs, returns):
            self._vf_optimizer.zero_grad()
            loss = self._value_function.compute_loss(*dataset)
            loss.backward()
            self._vf_optimizer.step()

    def _compute_loss_with_adv(self, obs, actions, advantages):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            actions (torch.Tensor): Actions fed to the environment
                with shape :math:`(N \dot [T], A*)`.
            advantages (torch.Tensor): Advantage value at each step
                with shape :math:`(N \dot [T], )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of objective.

        """
        log_likelihoods = self.policy(obs)[0].log_prob(actions)
        objectives = log_likelihoods * advantages

        if self._entropy_regularzied:
            policy_entropies = self._compute_policy_entropy(obs)
            objectives += self._policy_ent_coeff * policy_entropies

        return -objectives.mean()

    def _compute_policy_entropy(self, obs):
        r"""Compute entropy value of probability distribution.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N, P, O*)`.

        Returns:
            torch.Tensor: Calculated entropy values given observation
                with shape :math:`(N, P)`.

        """
        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = self.policy(obs)[0].entropy()
        else:
            policy_entropy = self.policy(obs)[0].entropy()

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _compute_bootstrapped_advantage(self, returns, valids, baselines):
        r"""Compute mean value of loss.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            rewards (torch.Tensor): Acquired rewards
                with shape :math:`(N, P)`.
            valids (list[int]): Numbers of valid steps in each episode
            baselines (torch.Tensor): Value function estimation at each step
                with shape :math:`(N, P)`.

        Returns:
            torch.Tensor: Calculated advantage values given rewards and
                baselines with shape :math:`(N \dot [T], )`.

        """

        advantages = returns - baselines
        advantage_flat = torch.cat(filter_valids(advantages, valids))

        if self._center_adv:
            means = advantage_flat.mean()
            variance = advantage_flat.var()
            advantage_flat = (advantage_flat - means) / (variance + 1e-8)

        if self._positive_adv:
            advantage_flat -= advantage_flat.min()

        return advantage_flat

    def process_samples(self, paths):
        r"""Process sample data based on the collected paths.

        Notes: P is the maximum episode length (self.max_episode_length)

        Args:
            paths (list[dict]): A list of collected paths

        Returns:
            torch.Tensor: The observations of the environment
                with shape :math:`(N, P, O*)`.
            torch.Tensor: The actions fed to the environment
                with shape :math:`(N, P, A*)`.
            torch.Tensor: The acquired rewards with shape :math:`(N, P)`.
            list[int]: Numbers of valid steps in each paths.
            torch.Tensor: Value function estimation at each step
                with shape :math:`(N, P)`.

        """

        valids = torch.Tensor([len(path['actions']) for path in paths]).int()

        obs = torch.stack([
            pad_to_last(path['observations'],
                        total_length=self.max_episode_length,
                        axis=0) for path in paths
        ])

        actions = torch.stack([
            pad_to_last(path['actions'],
                        total_length=self.max_episode_length,
                        axis=0) for path in paths
        ])

        with torch.no_grad():
            baselines = self._value_function(obs)

        augmented_rewards = []

        for idx, path in enumerate(paths):
            if path['step_types'][-1] != StepType.TERMINAL:
                augmented_rewards.append(
                    np.append(path['rewards'],
                              baselines[idx][len(path['rewards']) - 1]))
            else:
                augmented_rewards.append(path['rewards'])

        rewards = torch.stack([
            pad_to_last(path['rewards'], total_length=self.max_episode_length)
            for path in paths
        ])

        discounted = []

        for path, aug_rewards in zip(paths, augmented_rewards):
            if path['step_types'][-1] != StepType.TERMINAL:
                discounted.append(
                    pad_to_last(tu.discount_cumsum(aug_rewards,
                                                   self.discount)[:-1].copy(),
                                total_length=self.max_episode_length))
            else:
                discounted.append(
                    pad_to_last(tu.discount_cumsum(aug_rewards,
                                                   self.discount).copy(),
                                total_length=self.max_episode_length))

        returns = torch.stack(discounted)

        with torch.no_grad():
            baselines = self._value_function(obs)

        return obs, actions, rewards, returns, valids, baselines
