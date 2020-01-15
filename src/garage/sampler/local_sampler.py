"""Sampler that runs workers in the main process."""

from garage import TrajectoryBatch
from garage.sampler.sampler import Sampler


class LocalSampler(Sampler):
    """Sampler that runs workers in the main process.

    This is probably the simplest possible sampler. It's called the "Local"
    sampler because it runs everything in the same process and thread as where
    it was called from.

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

    """

    def __init__(self, worker_factory, agents, envs):
        # pylint: disable=super-init-not-called
        self._factory = worker_factory
        self._agents = worker_factory.prepare_worker_messages(agents)
        self._envs = worker_factory.prepare_worker_messages(envs)
        self._workers = [
            worker_factory(i) for i in range(worker_factory.n_workers)
        ]
        for worker, agent, env in zip(self._workers, self._agents, self._envs):
            worker.agent = agent
            worker.env = env

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
        return cls(worker_factory, agents, envs)

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
        agent_updates = self._factory.prepare_worker_messages(agent_update)
        env_updates = self._factory.prepare_worker_messages(env_update)
        for worker, agent_up, env_up in zip(self._workers, agent_updates,
                                            env_updates):
            worker.update_agent(agent_up)
            worker.update_env(env_up)
        batches = []
        completed_samples = 0
        while True:
            for worker in self._workers:
                batch = worker.rollout()
                completed_samples += len(batch.actions)
                batches.append(batch)
                if completed_samples > num_samples:
                    return TrajectoryBatch.concatenate(*batches)

    def shutdown_worker(self):
        """Shutdown the workers."""
        for worker in self._workers:
            worker.shutdown()
