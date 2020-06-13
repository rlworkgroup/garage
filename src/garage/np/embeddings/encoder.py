"""Base class for context encoder."""
import abc


class Encoder(abc.ABC):
    """Base class of context encoders for training meta-RL algorithms."""

    @property
    @abc.abstractmethod
    def spec(self):
        """garage.InOutSpec: Input and output space."""

    @property
    @abc.abstractmethod
    def input_dim(self):
        """int: Dimension of the encoder input."""

    @property
    @abc.abstractmethod
    def output_dim(self):
        """int: Dimension of the encoder output (embedding)."""

    def reset(self, do_resets=None):
        """Reset the encoder.

        This is effective only to recurrent encoder. do_resets is effective
        only to vectoried encoder.

        For a vectorized encoder, do_resets is an array of boolean indicating
        which internal states to be reset. The length of do_resets should be
        equal to the length of inputs.

        Args:
            do_resets (numpy.ndarray): Bool array indicating which states
                to be reset.

        """


class StochasticEncoder(Encoder):
    """An stochastic context encoders.

    An stochastic encoder maps an input to a distribution, but not a
    deterministic vector.

    """

    @property
    @abc.abstractmethod
    def distribution(self):
        """object: Embedding distribution."""
