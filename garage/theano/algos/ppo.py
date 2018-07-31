from garage.core import Serializable
from garage.theano.algos import NPO
from garage.theano.optimizers import PenaltyLbfgsOptimizer


class PPO(NPO, Serializable):
    """
    Penalized Policy Optimization.
    """

    def __init__(self, optimizer=None, optimizer_args=None, **kwargs):
        Serializable.quick_init(self, locals())
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
        super(PPO, self).__init__(optimizer=optimizer, **kwargs)
