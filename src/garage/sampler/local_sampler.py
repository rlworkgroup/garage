"""Sampler that runs workers in the main process."""
import copy
import click

from garage import TrajectoryBatch
from garage.sampler.sampler import Sampler
from garage.sampler.worker_factory import WorkerFactory
from garage.experiment.deterministic import get_seed


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

    def __init__(self, agents, envs, max_path_length=None,
                 worker_factory=None):
        # pylint: disable=super-init-not-called
        if worker_factory is None:
            assert max_path_length is not None
            worker_factory = WorkerFactory(seed=get_seed(),
                                           max_path_length=max_path_length)
        self._factory = worker_factory
        self._agents = worker_factory.prepare_worker_messages(agents)
        self._envs = worker_factory.prepare_worker_messages(
            envs, preprocess=copy.deepcopy)
        self._workers = [
            worker_factory(i) for i in range(worker_factory.n_workers)
        ]
        for worker, agent, env in zip(self._workers, self._agents, self._envs):
            worker.update_agent(agent)
            worker.update_env(env)
        self.total_env_steps = 0

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
        return cls(agents, envs, worker_factory=worker_factory)

    def _update_workers(self, agent_update, env_update):
        """Apply updates to the workers.

        Args:
            agent_update(object): Value which will be passed into the
                `agent_update_fn` before doing rollouts. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update(object): Value which will be passed into the
                `env_update_fn` before doing rollouts. If a list is passed in,
                it must have length exactly `factory.n_workers`, and will be
                spread across the workers.

        """
        agent_updates = self._factory.prepare_worker_messages(agent_update)
        env_updates = self._factory.prepare_worker_messages(
            env_update, preprocess=copy.deepcopy)
        for worker, agent_up, env_up in zip(self._workers, agent_updates,
                                            env_updates):
            worker.update_agent(agent_up)
            worker.update_env(env_up)

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
        self._update_workers(agent_update, env_update)
        batches = []
        completed_samples = 0
        with click.progressbar(length=num_samples, label='Sampling') as pbar:
            while True:
                for worker in self._workers:
                    batch = worker.rollout()
                    completed_samples += len(batch.actions)
                    batches.append(batch)
                    pbar.update(completed_samples)
                    if completed_samples >= num_samples:
                        samples = TrajectoryBatch.concatenate(*batches)
                        self.total_env_steps += sum(samples.lengths)
                        return samples

    def obtain_exact_trajectories(self,
                                  n_traj_per_worker,
                                  agent_update,
                                  env_update=None):
        """Sample an exact number of trajectories per worker.

        Args:
            n_traj_per_worker (int): Exact number of trajectories to gather for
                each worker.
            agent_update(object): Value which will be passed into the
                `agent_update_fn` before doing rollouts. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update(object): Value which will be passed into the
                `env_update_fn` before doing rollouts. If a list is passed in,
                it must have length exactly `factory.n_workers`, and will be
                spread across the workers.

        Returns:
            TrajectoryBatch: Batch of gathered trajectories. Always in worker
                order. In other words, first all trajectories from worker 0,
                then all trajectories from worker 1, etc.

        """
        self._update_workers(agent_update, env_update)
        batches = []
        for worker in self._workers:
            for _ in range(n_traj_per_worker):
                batch = worker.rollout()
                batches.append(batch)
        samples = TrajectoryBatch.concatenate(*batches)
        self.total_env_steps += sum(samples.lengths)
        return samples

    def shutdown_worker(self):
        """Shutdown the workers."""
        for worker in self._workers:
            worker.shutdown()

    def __getstate__(self):
        """Get the pickle state.

        Returns:
            dict: The pickled state.

        """
        state = self.__dict__.copy()
        # Workers aren't picklable (but WorkerFactory is).
        state['_workers'] = None
        return state

    def __setstate__(self, state):
        """Unpickle the state.

        Args:
            state (dict): Unpickled state.

        """
        self.__dict__.update(state)
        self._workers = [
            self._factory(i) for i in range(self._factory.n_workers)
        ]
        for worker, agent, env in zip(self._workers, self._agents, self._envs):
            worker.update_agent(agent)
            worker.update_env(env)
