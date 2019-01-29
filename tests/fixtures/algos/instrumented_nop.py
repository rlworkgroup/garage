"""
This file is a copy of garage/algos/nop.py
The only difference is the use of InstrumentedBatchPolopt to notify the test of
the different stages in the experiment lifecycle.
"""

from garage.misc.overrides import overrides
from tests.fixtures.algos.instrumented_batch_polopt import (
    InstrumentedBatchPolopt)


class InstrumentedNOP(InstrumentedBatchPolopt):
    """
    NOP (no optimization performed) policy search algorithm
    """

    def __init__(self, **kwargs):
        super(InstrumentedNOP, self).__init__(**kwargs)

    @overrides
    def init_opt(self):
        pass

    @overrides
    def optimize_policy(self, itr, samples_data):
        pass

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict()
