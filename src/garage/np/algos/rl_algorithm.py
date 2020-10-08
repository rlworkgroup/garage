"""Interface of RLAlgorithm."""
import abc


class RLAlgorithm(abc.ABC):
    """Base class for all the algorithms.

    Note:
        If the field sampler_cls exists, it will be by Trainer.setup to
        initialize a sampler.

    """

    # pylint: disable=too-few-public-methods

    @abc.abstractmethod
    def train(self, trainer):
        """Obtain samplers and start actual training for each epoch.

        Args:
            trainer (Trainer): Trainer is passed to give algorithm
                the access to trainer.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
