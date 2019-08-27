"""Interface of RLAlgorithm."""
import abc


class RLAlgorithm(abc.ABC):
    """Base class for all the algorithms.

    Note:
        If sampler_cls isn't specified to the LocalRunner,
        self.sampler_cls is required to provide default sampler
        for algorithm.

    """

    @abc.abstractmethod
    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
        pass

    @abc.abstractmethod
    def train(self, runner, batch_size):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.
            batch_size (int): Batch size used to obtain samplers.

        Returns:
            The average return in last epoch cycle.

        """
        pass
