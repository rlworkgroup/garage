"""Copy of NOP (no optimization performed) policy search algorithm.
This file is a copy of garage/algos/nop.py
The only difference is the use of InstrumentedBatchPolopt to notify the test of
the different stages in the experiment lifecycle.
"""

from tests.fixtures.algos.instrumented_batch_polopt import (
    InstrumentedBatchPolopt)


class InstrumentedNOP(InstrumentedBatchPolopt):
    """NOP (no optimization performed) policy search algorithm."""

    def init_opt(self):
        """Initialize the optimization procedure."""

    def optimize_policy(self, itr, paths):
        """Optimize the policy using the samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
