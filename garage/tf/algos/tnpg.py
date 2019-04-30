"""Truncated Natural Policy Gradient."""
from garage.tf.algos.npo import NPO
from garage.tf.optimizers import ConjugateGradientOptimizer


class TNPG(NPO):
    """
    Truncated Natural Policy Gradient.

    TNPG uses Conjugate Gradient to compute the policy gradient.

    """

    def __init__(self, optimizer=None, optimizer_args=None, **kwargs):
        if optimizer is None:
            optimizer = ConjugateGradientOptimizer
            default_args = dict(max_backtracks=1)
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
        super().__init__(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            name='TNPG',
            **kwargs)
