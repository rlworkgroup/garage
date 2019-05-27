from garage.tf.algos.npo import NPO, PGLoss
from garage.tf.optimizers import ConjugateGradientOptimizer


class TNPG(NPO):
    """Truncated Natural Policy Gradient.

    TNPG uses Conjugate Gradient to compute the policy gradient.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
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
        pg_loss (str): Objective.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies.
        max_kl_step (float): The maximum KL divergence between old and new
            policies.
        optimizer (float): The optimizer of the algorithm.
        optimizer_args (dict): Optimizer args.
        policy_ent_coeff (float): The coefficient of the policy entropy.
        use_softplus_entropy (bool): Whether to use softplus entropy.
        stop_entropy_gradient (bool): Whether to stop entropy gradient.
        name (str): The name of the algorithm.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=0.98,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss=PGLoss.SURROGATE,
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 name='TNPG'):
        if optimizer is None:
            optimizer = ConjugateGradientOptimizer
            default_args = dict(max_backtracks=1)
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
        super(TNPG, self).__init__(
            env_spec=env_spec,
            policy=policy,
            baseline=baseline,
            max_path_length=max_path_length,
            discount=discount,
            gae_lambda=gae_lambda,
            center_adv=center_adv,
            positive_adv=positive_adv,
            fixed_horizon=fixed_horizon,
            pg_loss=pg_loss,
            lr_clip_range=lr_clip_range,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            policy_ent_coeff=policy_ent_coeff,
            max_kl_step=max_kl_step,
            use_softplus_entropy=use_softplus_entropy,
            use_neg_logli_entropy=use_neg_logli_entropy,
            stop_entropy_gradient=stop_entropy_gradient,
            name=name)
