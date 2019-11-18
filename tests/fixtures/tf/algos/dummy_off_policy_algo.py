"""A dummy off-policy algorithm."""
from garage.np.algos import OffPolicyRLAlgorithm


class DummyOffPolicyAlgo(OffPolicyRLAlgorithm):
    """A dummy off-policy algorithm."""

    def init_opt(self):
        """Initialize the optimization procedure."""

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """

    def optimize_policy(self, itr, samples_data):
        """Optimize the policy using the samples.

        Args:
            itr (int): Iteration number.
            samples_data (dict): Processed sample data.
                See process_samples() for details.

        """
