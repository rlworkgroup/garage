"""Trust Region Policy Optimization for RL2."""
from garage.tf.algos import RL2
from garage.tf.optimizers import (ConjugateGradientOptimizer,
                                  PenaltyLbfgsOptimizer)


class RL2TRPO(RL2):
    """Trust Region Policy Optimization specific for RL^2.

    See https://arxiv.org/abs/1502.05477.

    Args:
        rl2_max_episode_length (int): Maximum length for episodes with respect
            to RL^2. Notice that it is different from the maximum episode
            length for the inner algorithm.
        meta_batch_size (int): Meta batch size.
        task_sampler (TaskSampler): Task sampler.
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_episode_length (int): Maximum length of a single episode.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies, as in PPO.
        max_kl_step (float): The maximum KL divergence between old and new
            policies, as in TRPO.
        optimizer (object): The optimizer of the algorithm. Should be the
            optimizers in garage.tf.optimizers.
        optimizer_args (dict): The arguments of the optimizer.
        policy_ent_coeff (float): The coefficient of the policy entropy.
            Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        kl_constraint (str): KL constraint, either 'hard' or 'soft'.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        meta_evaluator (MetaEvaluator): Evaluator for meta-RL algorithms.
        n_epochs_per_eval (int): If meta_evaluator is passed, meta-evaluation
            will be performed every `n_epochs_per_eval` epochs.
        name (str): The name of the algorithm.

    """

    def __init__(self,
                 rl2_max_episode_length,
                 meta_batch_size,
                 task_sampler,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_episode_length=500,
                 discount=0.99,
                 gae_lambda=0.98,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 kl_constraint='hard',
                 entropy_method='no_entropy',
                 meta_evaluator=None,
                 n_epochs_per_eval=10,
                 name='TRPO'):
        if not optimizer:
            if kl_constraint == 'hard':
                optimizer = ConjugateGradientOptimizer
            elif kl_constraint == 'soft':
                optimizer = PenaltyLbfgsOptimizer
            else:
                raise ValueError('Invalid kl_constraint')

        if optimizer_args is None:
            optimizer_args = dict()

        super().__init__(rl2_max_episode_length=rl2_max_episode_length,
                         meta_batch_size=meta_batch_size,
                         task_sampler=task_sampler,
                         env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         scope=scope,
                         max_episode_length=max_episode_length,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         pg_loss='surrogate',
                         lr_clip_range=lr_clip_range,
                         max_kl_step=max_kl_step,
                         optimizer=optimizer,
                         optimizer_args=optimizer_args,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         use_neg_logli_entropy=use_neg_logli_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method,
                         meta_evaluator=meta_evaluator,
                         n_epochs_per_eval=n_epochs_per_eval,
                         name=name)
