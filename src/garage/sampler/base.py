"""Base sampler class."""

import abc
import copy

from garage.sampler.config import SamplerConfig


class Sampler(abc.ABC):
    """Base class of all samplers."""

    def __init__(self, algo, env):
        """Construct a Sampler from an Algorithm.

        Using this method is deprecated.
        """
        self.algo = algo
        self.env = env

    @classmethod
    def construct(cls, config: SamplerConfig, agent, env) -> 'Sampler':
        """Construct this sampler from a config."""
        # This implementation works for most current implementations.
        # Relying on this implementation is deprecated, but this method is not.
        fake_algo = copy.copy(config)
        fake_algo.policy = agent
        return cls(fake_algo, env)

    def start_worker(self):
        """Initialize the sampler.

        i.e. launching parallel workers if necessary.

        This method is deprecated, please launch workers in construct instead.
        """

    @abc.abstractmethod
    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Collect at least a given number transitions (timesteps).

        Returns:
            list[dict]: A list of paths.

        """

    @abc.abstractmethod
    def shutdown_worker(self):
        """Terminate workers if necessary."""
