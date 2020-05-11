"""Interface of RLAlgorithm."""
import abc


class RLAlgorithm(abc.ABC):
    """Base class for all the algorithms.

    Note:
        If the field sampler_cls exists, it will be by LocalRunner.setup to
        initialize a sampler.

    """

    # pylint: disable=too-few-public-methods

    @abc.abstractmethod
    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
