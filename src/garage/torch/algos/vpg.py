"""Vanilla Policy Gradient (REINFORCE)."""
import collections

from dowel import tabular
import numpy as np
import torch
import torch.nn.functional as F

from garage.misc import tensor_utils
from garage.np.algos import BatchPolopt
from garage.torch.algos import loss_function_utils as loss_utils


class VPG(BatchPolopt):
    """Vanilla Policy Gradient (REINFORCE).

    VPG, also known as Reinforce, trains stochastic policy in an on-policy way.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.torch.policies.base.Policy): Policy.
        baseline (garage.np.baselines.Baseline): The baseline.
        max_path_length (int): Maximum length of a single rollout.
        policy_lr (float): Learning rate for training policy network.
        n_samples (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in torch.optim.
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
            baseline,
            max_path_length=500,
            policy_lr=1e-2,
            n_samples=1,
            discount=0.99,
            gae_lambda=1,
            center_adv=True,
            positive_adv=False,
            optimizer=None,
            policy_ent_coeff=0.0,
            use_softplus_entropy=False,
            stop_entropy_gradient=False,
            entropy_method='no_entropy',
    ):
        self._env_spec = env_spec
        self._policy_lr = policy_lr
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._policy_ent_coeff = policy_ent_coeff
        self._use_softplus_entropy = use_softplus_entropy
        self._stop_entropy_gradient = stop_entropy_gradient
        self._entropy_method = entropy_method
        self._eps = 1e-8

        self._maximum_entropy = (entropy_method == 'max')
        self._entropy_regularzied = (entropy_method == 'regularized')
        self._check_entropy_configuration(entropy_method, center_adv,
                                          stop_entropy_gradient,
                                          policy_ent_coeff)
        self._episode_reward_mean = collections.deque(maxlen=100)
        self._optimizer = optimizer(policy.parameters(), lr=policy_lr)

        super().__init__(policy=policy,
                         baseline=baseline,
                         discount=discount,
                         max_path_length=max_path_length,
                         n_samples=n_samples)

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

    def train_once(self, itr, paths):
        """Train the algorithm once.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            dict: Processed sample data, with key
                * average_return: (float)

        """
        samples_data = self.process_samples(itr, paths)

        self._optimize_policy(itr, paths)
        self.baseline.fit(paths)

        return samples_data['average_return']

    def _optimize_policy(self, itr, paths):
        """Optimize the policy with the paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        """
        # pylint: disable=unused-argument
        self._optimizer.zero_grad()

        policy_entropies = torch.stack([
            self._add_padding(
                self._get_policy_entropy(torch.Tensor(path['observations'])),
                self.max_path_length) for path in paths
        ])

        valids = [len(path['actions']) for path in paths]

        baselines = torch.stack([
            self._add_padding(self._get_baselines(path), self.max_path_length)
            for path in paths
        ])

        rewards = torch.stack([
            self._add_padding(torch.Tensor(path['rewards']),
                              self.max_path_length) for path in paths
        ])

        if self._maximum_entropy:
            rewards += self._policy_ent_coeff * policy_entropies

        advantages = loss_utils.compute_advantages(self.discount,
                                                   self._gae_lambda,
                                                   self.max_path_length,
                                                   baselines, rewards)

        if self._center_adv:
            means, variables = list(
                zip(*[(valid_adv.mean(), valid_adv.var())
                      for valid_adv in self._filter_valids(advantages, valids)]
                    ))
            advantages = F.batch_norm(advantages.t(),
                                      torch.Tensor(means),
                                      torch.Tensor(variables),
                                      eps=self._eps).t()

        if self._positive_adv:
            advantages -= advantages.min()

        log_likelihoods = torch.stack([
            self._add_padding(
                self.policy.log_likelihood(torch.Tensor(path['observations']),
                                           torch.Tensor(path['actions'])),
                self.max_path_length) for path in paths
        ])

        outputs = log_likelihoods * advantages

        if self._entropy_regularzied:
            outputs += self._policy_ent_coeff * policy_entropies

        loss = torch.cat(self._filter_valids(outputs, valids))
        # using a negative because optimizers use gradient descent,  whilst we
        # want gradient ascent.
        (-loss).mean().backward()

        self._optimizer.step()

    @staticmethod
    def _add_padding(tensor, length):
        """Add padding to the tensor.

        Args:
            tensor (torch.Tensor): target tensor to be padded
            length (int): length of padding to be added

        Returns:
            torch.Tensor: tensor with padding

        """
        padding_length = max(length - tensor.shape[-1], 0)
        return F.pad(tensor, (0, padding_length))

    def _get_policy_entropy(self, obs):
        """Get policy entropy of the observations.

        Args:
            obs (torch.Tensor): A 1D vector of per-step observation with shape
                (T), where T is the path length experienced by the agent.

        Returns:
            torch.Tensor: A 1D vector of calculated entropy with shape(T),
                where T is the path length experienced by the agent.

        """
        if self._stop_entropy_gradient:
            with torch.no_grad():
                policy_entropy = self.policy.get_entropy(obs)
        else:
            policy_entropy = self.policy.get_entropy(obs)

        # This prevents entropy from becoming negative for small policy std
        if self._use_softplus_entropy:
            policy_entropy = F.softplus(policy_entropy)

        return policy_entropy

    def _get_baselines(self, path):
        """Get baseline values of the path.

        Args:
            path (dict): collected path experienced by the agent

        Returns:
            torch.Tensor: A 2D vector of calculated baseline with shape(T),
                where T is the path length experienced by the agent.

        """
        if hasattr(self.baseline, 'predict_n'):
            return torch.Tensor(self.baseline.predict_n(path))
        return torch.Tensor(self.baseline.predict(path))

    @staticmethod
    def _filter_valids(tensor, valids):
        """Filter only valid tensor.

        Args:
            tensor (torch.Tensor): target tensor to be filtered
            valids (list[int]): length of the valid values for each path

        Returns:
            torch.Tensor: filtered valid tensors

        """
        return [tensor[i][:valids[i]] for i in range(len(valids))]

    def process_samples(self, itr, paths):
        """Process sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            dict: Processed sample data, with key
                * average_return: (float)

        """
        for path in paths:
            path['returns'] = tensor_utils.discount_cumsum(
                path['rewards'], self.discount)
        average_discounted_return = (np.mean(
            [path['returns'][0] for path in paths]))
        undiscounted_returns = [sum(path['rewards']) for path in paths]
        average_return = np.mean(undiscounted_returns)

        self._episode_reward_mean.extend(undiscounted_returns)

        tabular.record('Iteration', itr)
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', average_return)
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._episode_reward_mean))
        tabular.record('NumTrajs', len(paths))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))

        return dict(average_return=average_return)
