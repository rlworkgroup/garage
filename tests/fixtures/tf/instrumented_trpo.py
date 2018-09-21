"""This file is a copy of garage/tf/algos/trpo.py

The only difference is the use of InstrumentedNPO to notify the test of the
different stages in the experiment lifecycle.
"""

from enum import Enum
from enum import unique

from garage.tf.optimizers import ConjugateGradientOptimizer
from garage.tf.optimizers import PenaltyLbfgsOptimizer
from tests.fixtures.tf.instrumented_npo import InstrumentedNPO
from tests.fixtures.tf.instrumented_npo import PGLoss


@unique
class KLConstraint(Enum):
    HARD = "hard"
    SOFT = "soft"


class InstrumentedTRPO(InstrumentedNPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(self,
                 kl_constraint=KLConstraint.HARD,
                 optimizer=None,
                 optimizer_args=None,
                 **kwargs):
        if not optimizer:
            if kl_constraint == KLConstraint.HARD:
                optimizer = ConjugateGradientOptimizer
            elif kl_constraint == KLConstraint.SOFT:
                optimizer = PenaltyLbfgsOptimizer
            else:
                raise NotImplementedError("Unknown KLConstraint")

        if optimizer_args is None:
            optimizer_args = dict()

        super(InstrumentedTRPO, self).__init__(
            pg_loss=PGLoss.VANILLA,
            optimizer=optimizer,
            optimizer_args=optimizer_args,
            **kwargs)
