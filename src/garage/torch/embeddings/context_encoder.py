# pylint: disable=too-few-public-methods
"""Base context encoder class."""
import abc


class ContextEncoder(abc.ABC):
    """Base class of context encoders for training meta-RL algorithms."""

    def reset(self, num_tasks=1):
        """Reset hidden state task size.

        Args:
            num_tasks (int): Size of tasks.

        """
