"""Base class for context encoder."""
import abc


class Encoder(abc.ABC):
    """Base class of context encoders for training meta-RL algorithms."""

    @abc.abstractmethod
    def forward(self, input_value):
        """Encode an input value.

        Args:
            input_value (numpy.ndarray): Input values of (N, input_dim) shape.

        Returns:
            numpy.ndarray: Encoded embedding.

        """

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
        """scipy.stats.rv_generic: Embedding distribution."""

    def dist_info(self, input_value, state_infos):
        """Distribution info.

        Get the information of embedding distribution given an input.

        Args:
            input_value (np.ndarray): input values
            state_infos (dict): a dictionary whose values contain
                information about the predicted embedding given an input.

        """
