"""Base sampler class."""

import abc
import copy


class Sampler(abc.ABC):
    """Abstract base class of all samplers.

    Implementations of this class should override `construct`,
    `obtain_samples`, and `shutdown_worker`. `construct` takes a
    `WorkerFactory`, which implements most of the RL-specific functionality a
    `Sampler` needs. Specifically, it specifies how to construct `Worker`s,
    which know how to perform rollouts and update both agents and environments.

    Currently, `__init__` is also part of the interface, but calling it is
    deprecated. `start_worker` is also deprecated, and does not need to be
    implemented.
    """

    def __init__(self, algo, env):
        """Construct a Sampler from an Algorithm.

        Args:
            algo(garage.RLAlgorithm): The RL Algorithm controlling this
                sampler.
            env(gym.Env): The environment being sampled from.

        Calling this method is deprecated.

        """
        self.algo = algo
        self.env = env

    @classmethod
    def from_worker_factory(cls, worker_factory, agents, envs):
        """Construct this sampler.

        Args:
            worker_factory(WorkerFactory): Pickleable factory for creating
                workers. Should be transmitted to other processes / nodes where
                work needs to be done, then workers should be constructed
                there.
            agents(Agent or List[Agent]): Agent(s) to use to perform rollouts.
                If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.
            envs(gym.Env or List[gym.Env]): Environment rollouts are performed
                in. If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.

        Returns:
            Sampler: An instance of `cls`.

        """
        # This implementation works for most current implementations.
        # Relying on this implementation is deprecated, but calling this method
        # is not.
        fake_algo = copy.copy(worker_factory)
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
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update(object): Value which will be passed into the
                `env_update_fn` before doing rollouts. If a list is passed in,
                it must have length exactly `factory.n_workers`, and will be
                spread across the workers.

        Returns:
            garage.TrajectoryBatch: The batch of collected trajectories.

        """

    @abc.abstractmethod
    def shutdown_worker(self):
        """Terminate workers if necessary.

        Because Python object destruction can be somewhat unpredictable, this
        method isn't deprecated.
        """
