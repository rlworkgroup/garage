"""Advantage Actor-Critic (A2C)."""

from garage.torch.algos import VPG


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
