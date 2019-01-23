import abc


class Checkpointer(abc.ABC):
    def __init__(self, prefix, resume=True):
        self.resume = resume
        self.prefix = prefix

    @abc.abstractmethod
    def load(self, **kwargs):
        """Load from or initialize checkpoint.

        Args:
            **kwargs: named objects to save if no checkpoint exists.

        Returns:
            dict: named objects loaded from checkpoint.

        """
        raise NotImplementedError

    @abc.abstractmethod
    def save(self, **kwargs):
        """Save objects to a new checkpoint.

        Args:
            **kwargs: named objects to save.

        """
        raise NotImplementedError
