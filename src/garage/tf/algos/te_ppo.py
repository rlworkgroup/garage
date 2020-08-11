"""Proximal Policy Optimization with Task Embedding."""
from garage.tf.algos.te_npo import TENPO
from garage.tf.optimizers import FirstOrderOptimizer


class TEPPO(TENPO):
    """Proximal Policy Optimization with Task Embedding.

    See https://karolhausman.github.io/pdf/hausman17nips-ws2.pdf for algorithm
    reference.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.TaskEmbeddingPolicy): Policy.
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
        encoder_ent_coeff (float): The coefficient of the policy encoder
            entropy. Setting it to zero would mean no entropy regularization.
        use_softplus_entropy (bool): Whether to estimate the softmax
            distribution of the entropy to prevent the entropy from being
            negative.
        stop_ce_gradient (bool): Whether to stop the cross entropy gradient.
        inference (garage.tf.embedding.encoder.StochasticEncoder): A encoder
            that infers the task embedding from a state trajectory.
        inference_optimizer (object): The optimizer of the inference. Should be
            an optimizer in garage.tf.optimizers.
        inference_optimizer_args (dict): The arguments of the inference
            optimizer.
        inference_ce_coeff (float): The coefficient of the cross entropy of
            task embeddings inferred from task one-hot and state trajectory.
            This is effectively the coefficient of log-prob of inference.
        name (str): The name of the algorithm.

    """

    def __init__(self,
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
                 policy_ent_coeff=1e-3,
                 encoder_ent_coeff=1e-3,
                 use_softplus_entropy=False,
                 stop_ce_gradient=False,
                 inference=None,
                 inference_optimizer=None,
                 inference_optimizer_args=None,
                 inference_ce_coeff=1e-3,
                 name='PPOTaskEmbedding'):

        optimizer = optimizer or FirstOrderOptimizer
        optimizer_args = optimizer_args or dict(batch_size=32,
                                                max_episode_length=10)

        inference_optimizer = inference_optimizer or FirstOrderOptimizer
        inference_optimizer_args = inference_optimizer_args or dict(
            batch_size=32, max_episode_length=10)

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         scope=scope,
                         max_episode_length=max_episode_length,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         lr_clip_range=lr_clip_range,
                         max_kl_step=max_kl_step,
                         optimizer=optimizer,
                         optimizer_args=optimizer_args,
                         policy_ent_coeff=policy_ent_coeff,
                         encoder_ent_coeff=encoder_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_ce_gradient=stop_ce_gradient,
                         inference=inference,
                         inference_optimizer=inference_optimizer,
                         inference_optimizer_args=inference_optimizer_args,
                         inference_ce_coeff=inference_ce_coeff,
                         name=name)
