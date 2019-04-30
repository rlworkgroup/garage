"""Episodic Reward Weighted Regression."""
from garage.tf.algos.vpg import VPG
from garage.tf.optimizers import LbfgsOptimizer


class ERWR(VPG):
    """
    Episodic Reward Weighted Regression [1].

    Notes
    -----
    This does not implement the original RwR [2]_ that deals with "immediate
    reward problems" since it doesn't find solutions that optimize for
    temporally delayed rewards.

    .. [1] Kober, Jens, and Jan R. Peters. "Policy search for motor primitives
           in robotics." Advances in neural information processing systems.
           2009.
    .. [2] Peters, Jan, and Stefan Schaal. "Using reward-weighted regression
           for reinforcement learning of task space control." Approximate
           Dynamic Programming and Reinforcement Learning, 2007. ADPRL 2007.
           IEEE International Symposium on. IEEE, 2007.

    """

    def __init__(self,
                 optimizer=None,
                 optimizer_args=None,
                 positive_adv=True,
                 **kwargs):
        if optimizer is None:
            optimizer = LbfgsOptimizer
            if optimizer_args is None:
                optimizer_args = dict()
        super(ERWR, self).__init__(
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            positive_adv=positive_adv,
            **kwargs)
