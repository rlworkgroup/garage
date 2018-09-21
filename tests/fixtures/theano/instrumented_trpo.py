"""This file is a copy of garage/theano/algos/trpo.py

The only difference is the use of InstrumentedNPO to notify the test of the
different stages in the experiment lifecycle.
"""

from garage.theano.optimizers import ConjugateGradientOptimizer
from tests.fixtures.theano.instrumented_npo import InstrumentedNPO


class InstrumentedTRPO(InstrumentedNPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(self, optimizer=None, optimizer_args=None, **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(InstrumentedTRPO, self).__init__(optimizer=optimizer, **kwargs)
