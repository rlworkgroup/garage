from garage.misc import ext
from garage.optimizers import ConjugateGradientOptimizer
from garage.theano.algos import NPO


class TNPG(NPO):
    """
    Truncated Natural Policy Gradient.
    """

    def __init__(self, optimizer=None, optimizer_args=None, **kwargs):
        if optimizer is None:
            default_args = dict(max_backtracks=1)
            if optimizer_args is None:
                optimizer_args = default_args
            else:
                optimizer_args = dict(default_args, **optimizer_args)
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(TNPG, self).__init__(optimizer=optimizer, **kwargs)
