"""Trust Region Policy Optimization."""
import torch

from garage.torch.algos import VPG
from garage.torch.optimizers import ConjugateGradientOptimizer


class TRPO(VPG):
    """Trust Region Policy Optimization (TRPO).

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.torch.policies.base.Policy): Policy.
        baseline (garage.np.baselines.Baseline): The baseline.
        max_path_length (int): Maximum length of a single rollout.
        policy_lr (float): Learning rate for training policy network.
        num_train_per_epoch (int): Number of train_once calls per epoch.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        max_kl (float): Maximum KL divergence between old and new policies.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in torch.optim.
        optimizer_args (dict): Arguments required to initialize the optimizer.
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
                 max_path_length=100,
                 policy_lr=3e-4,
                 num_train_per_epoch=1,
                 discount=0.99,
                 gae_lambda=0.98,
                 max_kl=0.01,
                 center_adv=True,
                 positive_adv=False,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy'):
        if optimizer is None:
            optimizer = ConjugateGradientOptimizer
            optimizer_args = {'max_constraint_value': max_kl}

        super().__init__(env_spec, policy, baseline, max_path_length,
                         policy_lr, num_train_per_epoch, discount, gae_lambda,
                         center_adv, positive_adv, optimizer, optimizer_args,
                         policy_ent_coeff, use_softplus_entropy,
                         stop_entropy_gradient, entropy_method)

        self._kl = None

    def _compute_objective(self, advantages, valids, obs, actions, rewards):
        """Compute the surrogate objective.

        Args:
            advantages (torch.Tensor): Expected rewards over the actions
            valids (list[int]): length of the valid values for each path
            obs (torch.Tensor): Observation from the environment.
            actions (torch.Tensor): Predicted action.
            rewards (torch.Tensor): Feedback from the environment.

        Returns:
            torch.Tensor: Calculated objective values

        """
        with torch.no_grad():
            old_ll = self._old_policy.log_likelihood(obs, actions)

        new_ll = self.policy.log_likelihood(obs, actions)
        likelihood_ratio = (new_ll - old_ll).exp()

        # Calculate surrogate
        surrogate = likelihood_ratio * advantages

        return surrogate

    def _optimize(self, itr, paths, valids, obs, actions, rewards):
        self._optimizer.step(
            f_loss=lambda: self._compute_loss(itr, paths, valids, obs, actions,
                                              rewards),
            f_constraint=lambda: self._compute_kl_constraint(obs))
