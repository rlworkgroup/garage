"""Advantage Actor-Critic (A2C)."""
from garage.tf.algos import VPG


class A2C(VPG):
    """A2C paper: https://arxiv.org/abs/1602.01783.

    A2C is the synchronous variant of the A3C algorithm described
    in the paper. The primary difference between A2C and A3C is that
    the former uses one policy and baseline network and
    synchronously updates these networks after all workers complete
    the sampling step.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_episode_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the vanilla objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        name (str): The name of the algorithm.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_episode_length=500,
                 discount=0.99,
                 center_adv=True,
                 positive_adv=False,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.01,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='regularized',
                 name='A2C'):

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         scope=scope,
                         max_episode_length=max_episode_length,
                         discount=discount,
                         gae_lambda=1,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         optimizer=optimizer,
                         optimizer_args=optimizer_args,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         use_neg_logli_entropy=False,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         name=name)
