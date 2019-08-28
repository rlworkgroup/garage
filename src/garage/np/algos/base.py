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
    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            The average return in last epoch cycle or None.

        """
        pass
