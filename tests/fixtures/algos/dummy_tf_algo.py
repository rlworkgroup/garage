"""Dummy algorithm."""
from garage.np.algos import RLAlgorithm


class DummyTFAlgo(RLAlgorithm):
    """Dummy algorithm."""

    def init_opt(self):
        """Initialize the optimization procedure.

        If using tensorflow, this may include declaring all the variables and
        compiling functions.

        """

    def optimize_policy(self, samples_data):
        """Optimize the policy using the samples.

        Args:
            samples_data (dict): Processed sample data.
                See garage.tf.paths_to_tensors() for details.

        """
