"""Model-Agnostic Meta-Learning (MAML) algorithm applied to PPO."""
import torch

from garage import _Default
from garage.torch.algos import PPO
from garage.torch.algos.maml import MAML
from garage.torch.optimizers import OptimizerWrapper


class MAMLPPO(MAML):
    """Model-Agnostic Meta-Learning (MAML) applied to PPO.

    Args:
        env (Environment): A multi-task environment.
        policy (garage.torch.policies.Policy): Policy.
        value_function (garage.np.baselines.Baseline): The value function.
        inner_lr (float): Adaptation learning rate.
        outer_lr (float): Meta policy learning rate.
        lr_clip_range (float): The limit on the likelihood ratio between
            policies.
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
        meta_batch_size (int): Number of tasks sampled per batch.
        num_grad_updates (int): Number of adaptation gradient steps.
        meta_evaluator (garage.experiment.MetaEvaluator): A meta evaluator for
            meta-testing. If None, don't do meta-testing.
        evaluate_every_n_epochs (int): Do meta-testing every this epochs.

    """

    def __init__(self,
                 env,
                 policy,
                 value_function,
                 inner_lr=_Default(1e-1),
                 outer_lr=1e-3,
                 lr_clip_range=5e-1,
                 max_episode_length=100,
                 discount=0.99,
                 gae_lambda=1.0,
                 center_adv=True,
                 positive_adv=False,
                 policy_ent_coeff=0.0,
                 use_softplus_entropy=False,
                 stop_entropy_gradient=False,
                 entropy_method='no_entropy',
                 meta_batch_size=20,
                 num_grad_updates=1,
                 meta_evaluator=None,
                 evaluate_every_n_epochs=1):

        policy_optimizer = OptimizerWrapper(
            (torch.optim.Adam, dict(lr=inner_lr)), policy)
        vf_optimizer = OptimizerWrapper((torch.optim.Adam, dict(lr=inner_lr)),
                                        value_function)

        inner_algo = PPO(env.spec,
                         policy,
                         value_function,
                         policy_optimizer=policy_optimizer,
                         vf_optimizer=vf_optimizer,
                         lr_clip_range=lr_clip_range,
                         max_episode_length=max_episode_length,
                         num_train_per_epoch=1,
                         discount=discount,
                         gae_lambda=gae_lambda,
                         center_adv=center_adv,
                         positive_adv=positive_adv,
                         policy_ent_coeff=policy_ent_coeff,
                         use_softplus_entropy=use_softplus_entropy,
                         stop_entropy_gradient=stop_entropy_gradient,
                         entropy_method=entropy_method)

        super().__init__(inner_algo=inner_algo,
                         env=env,
                         policy=policy,
                         meta_optimizer=torch.optim.Adam,
                         meta_batch_size=meta_batch_size,
                         inner_lr=inner_lr,
                         outer_lr=outer_lr,
                         num_grad_updates=num_grad_updates,
                         meta_evaluator=meta_evaluator,
                         evaluate_every_n_epochs=evaluate_every_n_epochs)
