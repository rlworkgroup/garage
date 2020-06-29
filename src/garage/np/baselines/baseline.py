"""Base class for all baselines."""
import abc


class Baseline(abc.ABC):
    """Base class for all baselines."""

    @abc.abstractmethod
    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        """

    @abc.abstractmethod
    def predict(self, paths):
        """Predict value based on paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        Returns:
            numpy.ndarray: Predicted value.

        """
