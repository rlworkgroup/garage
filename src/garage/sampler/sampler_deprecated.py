"""Base class of Sampler."""
import abc


class Sampler(abc.ABC):
    """Sampler interface."""

    @abc.abstractmethod
    def start_worker(self):
        """Initialize the sampler.

        e.g. launching parallel workers if necessary.

        """

    @abc.abstractmethod
    def obtain_samples(self, itr, batch_size, whole_paths):
        """Collect samples for the given iteration number.

        Args:
            itr (int): Number of iteration.
            batch_size (int): Number of environment steps in one batch.
            whole_paths (bool): Whether to use whole path or truncated.

        Returns:
            list[dict]: A list of paths.

        """

    @abc.abstractmethod
    def shutdown_worker(self):
        """Terminate workers if necessary."""


class BaseSampler(Sampler):
    # pylint: disable=abstract-method
    """Base class for sampler.

    Args:
        algo (garage.np.algos.RLAlgorithm): The algorithm.
        env (gym.Env): The environment.

    """

    def __init__(self, algo, env):
        self.algo = algo
        self.env = env
