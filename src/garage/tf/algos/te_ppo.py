"""Proximal Policy Optimization with Task Embedding."""
from garage.tf.algos.te_npo import TENPO
from garage.tf.optimizers import FirstOrderOptimizer


class TEPPO(TENPO):
    """Proximal Policy Optimization with Task Embedding.

    See https://karolhausman.github.io/pdf/hausman17nips-ws2.pdf for algorithm
    reference.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.TaskEmbeddingPolicy): Policy.
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
        pg_loss (str): A string from: 'vanilla', 'surrogate',
            'surrogate_clip'. The type of loss functions to use.
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
        use_neg_logli_entropy (bool): Whether to estimate the entropy as the
            negative log likelihood of the action.
        stop_entropy_gradient (bool): Whether to stop the entropy gradient.
        stop_ce_gradient (bool): Whether to stop the cross entropy gradient.
        entropy_method (str): A string from: 'max', 'regularized',
            'no_entropy'. The type of entropy method to use. 'max' adds the
            dense entropy to the reward for each time step. 'regularized' adds
            the mean entropy to the surrogate objective. See
            https://arxiv.org/abs/1805.00909 for more details.
        flatten_input (bool): Whether to flatten input along the observation
            dimension. If True, for example, an observation with shape (2, 4)
            will be flattened to 8.
        inference (garage.tf.embedding.encoder.StochasticEncoder): A encoder
            that infers the task embedding from trajectory.
        inference_optimizer (object): The optimizer of the inference. Should be
            an optimizer in garage.tf.optimizers.
        inference_optimizer_args (dict): The arguments of the inference
            optimizer.
        inference_ce_coeff (float): The coefficient of the cross entropy of
            task embeddings inferred from task one-hot and trajectory. This is
            effectively the coefficient of log-prob of inference.
        name (str): The name of the algorithm.

    Note:
        sane defaults for entropy configuration:
            - entropy_method='max', center_adv=False, stop_gradient=True
              (center_adv normalizes the advantages tensor, which will
              significantly alleviate the effect of entropy. It is also
              recommended to turn off entropy gradient so that the agent
              will focus on high-entropy actions instead of increasing the
              variance of the distribution.)
            - entropy_method='regularized', stop_gradient=False,
              use_neg_logli_entropy=False

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=0.98,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 pg_loss='surrogate_clip',
                 lr_clip_range=0.01,
                 max_kl_step=0.01,
                 optimizer=None,
                 optimizer_args=None,
                 policy_ent_coeff=0.0,
                 encoder_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 use_neg_logli_entropy=False,
                 stop_entropy_gradient=False,
                 stop_ce_gradient=False,
                 entropy_method='no_entropy',
                 flatten_input=True,
                 inference=None,
                 inference_optimizer=None,
                 inference_optimizer_args=None,
                 inference_ce_coeff=0.0,
                 name='PPOTaskEmbedding'):
        optimizer, optimizer_args = self._build_optimizer(
            optimizer, optimizer_args)
        inference_optimizer, inference_optimizer_args = self._build_optimizer(
            inference_optimizer, inference_optimizer_args)

        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         scope=scope,
                         max_path_length=max_path_length,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         fixed_horizon=fixed_horizon,
                         pg_loss=pg_loss,
                         lr_clip_range=lr_clip_range,
                         max_kl_step=max_kl_step,
                         optimizer=optimizer,
                         optimizer_args=optimizer_args,
                         policy_ent_coeff=policy_ent_coeff,
                         encoder_ent_coeff=encoder_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         use_neg_logli_entropy=use_neg_logli_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         stop_ce_gradient=stop_ce_gradient,
                         entropy_method=entropy_method,
                         flatten_input=flatten_input,
                         inference=inference,
                         inference_optimizer=inference_optimizer,
                         inference_optimizer_args=inference_optimizer_args,
                         inference_ce_coeff=inference_ce_coeff,
                         name=name)

    def _build_optimizer(self, optimizer, optimizer_args):
        """Build up optimizer for policy.

        Args:
            optimizer (obj): Policy optimizer. Should be one of the optimizers
                in garage.tf.optimizers.
            optimizer_args (dict): The arguments of the optimizer.

        Returns:
            obj: Policy optimizer. Should be one of the optimizers
                in garage.tf.optimizers.
            dict: The arguments of the optimizer.

        """
        if optimizer is None:
            optimizer = FirstOrderOptimizer
        if optimizer_args is None:
            optimizer_args = dict(
                batch_size=32,
                max_epochs=10,
            )
        return optimizer, optimizer_args
