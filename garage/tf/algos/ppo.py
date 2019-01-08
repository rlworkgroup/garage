"""This module implements a PPO algorithm."""
from garage.tf.algos import NPO
from garage.tf.algos.npo import PGLoss
from garage.tf.optimizers import FirstOrderOptimizer


class PPO(NPO):
    """
    Proximal Policy Optimization.

    See https://arxiv.org/abs/1707.06347.
    """

    def __init__(self, optimizer=None, optimizer_args=None, **kwargs):
        """
        Construct class.

        :param optimizer: Optimizer.
        :param optimizer_args: Optimizer args.
        :param kwargs:
        """
        if optimizer is None:
            optimizer = FirstOrderOptimizer
            if optimizer_args is None:
                optimizer_args = dict()
        super(PPO, self).__init__(
            pg_loss=PGLoss.SURROGATE_CLIP,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            name="PPO",
            **kwargs)
