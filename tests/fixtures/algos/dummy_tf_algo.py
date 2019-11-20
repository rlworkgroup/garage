"""Dummy algorithm."""
from garage.tf.algos import BatchPolopt


class DummyTFAlgo(BatchPolopt):
    """Dummy algorithm."""

    def init_opt(self):
        """Initialize the optimization procedure.

        If using tensorflow, this may include declaring all the variables and
        compiling functions.

        """

    def optimize_policy(self, itr, samples_data):
        """Optimize the policy using the samples.

        Args:
            itr (int): Iteration number.
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        """
