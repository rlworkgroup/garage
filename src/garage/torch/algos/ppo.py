"""Proximal Policy Optimization (PPO)."""
import copy

import torch

from garage.torch.algos import VPG


class PPO(VPG):
    """Proximal Policy Optimization (PPO).

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.torch.policies.base.Policy): Policy.
        baseline (garage.np.baselines.Baseline): The baseline.
        max_path_length (int): Maximum length of a single rollout.
        policy_lr (float): Learning rate for training policy network.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies.
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

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 max_path_length=500,
                 policy_lr=3e-4,
                 lr_clip_range=2e-1,
                 n_samples=1,
                 discount=0.99,
                 gae_lambda=0.97,
                 center_adv=True,
                 positive_adv=False,
                 optimizer=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy'):
        super().__init__(env_spec, policy, baseline, max_path_length,
                         policy_lr, n_samples, discount, gae_lambda,
                         center_adv, positive_adv, optimizer, policy_ent_coeff,
                         use_softplus_entropy, stop_entropy_gradient,
                         entropy_method)

        self._lr_clip_range = lr_clip_range
        self._old_policy = copy.deepcopy(self.policy)

    def _compute_objective(self, advantages, valids, obs, actions, rewards):
        """Compute objective using surrogate value and clipped surrogate value.

        Args:
            advantages (torch.Tensor): Expected rewards over the actions
            valids (list[int]): length of the valid values for each path
            obs (torch.Tensor): Observation from the environment.
            actions (torch.Tensor): Predicted action.
            rewards (torch.Tensor): Feedback from the environment.

        Returns:
            torch.Tensor: Calculated objective values

        """
        # pylint: disable=unused-argument
        # Compute constraint
        with torch.no_grad():
            old_ll = self._old_policy.log_likelihood(obs, actions)
        new_ll = self.policy.log_likelihood(obs, actions)

        likelihood_ratio = (new_ll - old_ll).exp()

        # Memorize the policy state_dict
        self._old_policy.load_state_dict(self.policy.state_dict())

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        # Clipping the constraint
        likelihood_ratio_clip = torch.clamp(likelihood_ratio,
                                            min=1 - self._lr_clip_range,
                                            max=1 + self._lr_clip_range)

        # Calculate surrotate clip
        surrogate_clip = likelihood_ratio_clip * advantages

        return torch.min(surrogate, surrogate_clip)
