"""Base sampler class."""

import abc
import copy


class Sampler(abc.ABC):
    """Base class of all samplers."""

    def __init__(self, algo, env):
        """Construct a Sampler from an Algorithm.

        Calling this method is deprecated.
        """
        self.algo = algo
        self.env = env

    @classmethod
    def construct(cls, config, agents, envs):
        """Construct this sampler from a config.

        Args:
            config(SamplerConfig): Configuration which specifies how to
                intialize workers, update agents and environments, and perform
                rollouts.
            agents(Agent or [Agent]): Agent(s) to use to perform rollouts. It
                will be passed into `config.agent_update_fn` before doing any
                rollouts. If a list is passed in, it must have length exactly
                `config.n_workers`, and will be spread across the workers.
            envs(gym.Env or [gym.Env]): Environment rollouts are performed in.
                It will be passed into `config.env_update_fn` before doing any
                rollouts. If a list is passed in, it must have length exactly
                `config.n_workers`, and will be spread across the workers.

        Returns:
            Sampler: An instance of `cls`.

        """
        # This implementation works for most current implementations.
        # Relying on this implementation is deprecated, but calling this method
        # is not.
        fake_algo = copy.copy(config)
        fake_algo.policy = agents
        return cls(fake_algo, envs)

    def start_worker(self):
        """Initialize the sampler.

        i.e. launching parallel workers if necessary.

        This method is deprecated, please launch workers in construct instead.
        """

    @abc.abstractmethod
    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Collect at least a given number transitions (timesteps).

        Args:
            itr(int): The current iteration number. Using this argument is
                deprecated.
            num_samples(int): Minimum number of transitions / timesteps to
                sample.
            agent_update(object): Value which will be passed into the
                `agent_update_fn` before doing rollouts. If a list is passed
                in, it must have length exactly `config.n_workers`, and will be
                spread across the workers.
            env_update(object): Value which will be passed into the
                `env_update_fn` before doing rollouts. If a list is passed in,
                it must have length exactly `config.n_workers`, and will be
                spread across the workers.

        Returns:
            list[dict]: A list of paths.

        """

    @abc.abstractmethod
    def shutdown_worker(self):
        """Terminate workers if necessary."""
