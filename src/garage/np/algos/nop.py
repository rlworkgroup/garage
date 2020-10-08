"""NOP (no optimization performed) policy search algorithm."""
from garage.np.algos.rl_algorithm import RLAlgorithm


class NOP(RLAlgorithm):
    """NOP (no optimization performed) policy search algorithm."""

    def init_opt(self):
        """Initialize the optimization procedure."""

    def optimize_policy(self, paths):
        """Optimize the policy using the samples.

        Args:
            paths (list[dict]): A list of collected paths.

        """

    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Trainer is passed to give algorithm
                the access to trainer.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
