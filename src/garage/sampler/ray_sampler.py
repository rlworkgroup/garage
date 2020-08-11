"""This is an implementation of an on policy batch sampler.

Uses a data parallel design.
Included is a sampler that deploys sampler workers.
The sampler workers must implement some type of set agent parameters
function, and a rollout function.

"""
from collections import defaultdict
import itertools

import click
import cloudpickle
import ray

from garage import EpisodeBatch
from garage.sampler.sampler import Sampler


class RaySampler(Sampler):
    """Samples episodes in a data-parallel fashion using a Ray cluster.

    Args:
        worker_factory (WorkerFactory): Used for worker behavior.
        agents (list[Policy]): Agents to distribute across workers.
        envs (list[Environment]): Environments to distribute across workers.

    """

    def __init__(self, worker_factory, agents, envs):
        # pylint: disable=super-init-not-called
        if not ray.is_initialized():
            ray.init(log_to_driver=False)
        self._sampler_worker = ray.remote(SamplerWorker)
        self._worker_factory = worker_factory
        self._agents = agents
        self._envs = self._worker_factory.prepare_worker_messages(envs)
        self._all_workers = defaultdict(None)
        self._workers_started = False
        self.start_worker()

    @classmethod
    def from_worker_factory(cls, worker_factory, agents, envs):
        """Construct this sampler.

        Args:
            worker_factory (WorkerFactory): Pickleable factory for creating
                workers. Should be transmitted to other processes / nodes where
                work needs to be done, then workers should be constructed
                there.
            agents (Policy or List[Policy]): Agent(s) to use to sample
                episodes. If a list is passed in, it must have length exactly
                `worker_factory.n_workers`, and will be spread across the
                workers.
            envs (Environment or List[Environment]): Environment from which
                episodes are sampled. If a list is passed in, it must have
                length exactly `worker_factory.n_workers`, and will be spread
                across the workers.

        Returns:
            Sampler: An instance of `cls`.

        """
        return cls(worker_factory, agents, envs)

    def start_worker(self):
        """Initialize a new ray worker."""
        if self._workers_started:
            return
        self._workers_started = True
        # We need to pickle the agent so that we can e.g. set up the TF.Session
        # in the worker *before* unpickling it.
        agent_pkls = self._worker_factory.prepare_worker_messages(
            self._agents, cloudpickle.dumps)
        for worker_id in range(self._worker_factory.n_workers):
            self._all_workers[worker_id] = self._sampler_worker.remote(
                worker_id, self._envs[worker_id], agent_pkls[worker_id],
                self._worker_factory)

    def _update_workers(self, agent_update, env_update):
        """Update all of the workers.

        Args:
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling_episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            list[ray._raylet.ObjectID]: Remote values of worker ids.

        """
        updating_workers = []
        param_ids = self._worker_factory.prepare_worker_messages(
            agent_update, ray.put)
        env_ids = self._worker_factory.prepare_worker_messages(
            env_update, ray.put)
        for worker_id in range(self._worker_factory.n_workers):
            worker = self._all_workers[worker_id]
            updating_workers.append(
                worker.update.remote(param_ids[worker_id], env_ids[worker_id]))
        return updating_workers

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Sample the policy for new episodes.

        Args:
            itr (int): Iteration number.
            num_samples (int): Number of steps the the sampler should collect.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            EpisodeBatch: Batch of gathered episodes.

        """
        active_workers = []
        completed_samples = 0
        batches = []

        # update the policy params of each worker before sampling
        # for the current iteration
        idle_worker_ids = []
        updating_workers = self._update_workers(agent_update, env_update)

        with click.progressbar(length=num_samples, label='Sampling') as pbar:
            while completed_samples < num_samples:
                # if there are workers still being updated, check
                # which ones are still updating and take the workers that
                # are done updating, and start collecting episodes on those
                # workers.
                if updating_workers:
                    updated, updating_workers = ray.wait(updating_workers,
                                                         num_returns=1,
                                                         timeout=0.1)
                    upd = [ray.get(up) for up in updated]
                    idle_worker_ids.extend(upd)

                # if there are idle workers, use them to collect episodes and
                # mark the newly busy workers as active
                while idle_worker_ids:
                    idle_worker_id = idle_worker_ids.pop()
                    worker = self._all_workers[idle_worker_id]
                    active_workers.append(worker.rollout.remote())

                # check which workers are done/not done collecting a sample
                # if any are done, send them to process the collected
                # episode if they are not, keep checking if they are done
                ready, not_ready = ray.wait(active_workers,
                                            num_returns=1,
                                            timeout=0.001)
                active_workers = not_ready
                for result in ready:
                    ready_worker_id, episode_batch = ray.get(result)
                    idle_worker_ids.append(ready_worker_id)
                    num_returned_samples = episode_batch.lengths.sum()
                    completed_samples += num_returned_samples
                    batches.append(episode_batch)
                    pbar.update(num_returned_samples)

        return EpisodeBatch.concatenate(*batches)

    def obtain_exact_episodes(self,
                              n_eps_per_worker,
                              agent_update,
                              env_update=None):
        """Sample an exact number of episodes per worker.

        Args:
            n_eps_per_worker (int): Exact number of episodes to gather for
                each worker.
            agent_update (object): Value which will be passed into the
                `agent_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update (object): Value which will be passed into the
                `env_update_fn` before sampling episodes. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.

        Returns:
            EpisodeBatch: Batch of gathered episodes. Always in worker
                order. In other words, first all episodes from worker 0, then
                all episodes from worker 1, etc.

        """
        active_workers = []
        episodes = defaultdict(list)

        # update the policy params of each worker before sampling
        # for the current iteration
        idle_worker_ids = []
        updating_workers = self._update_workers(agent_update, env_update)

        with click.progressbar(length=self._worker_factory.n_workers,
                               label='Sampling') as pbar:
            while any(
                    len(episodes[i]) < n_eps_per_worker
                    for i in range(self._worker_factory.n_workers)):
                # if there are workers still being updated, check
                # which ones are still updating and take the workers that
                # are done updating, and start collecting episodes on
                # those workers.
                if updating_workers:
                    updated, updating_workers = ray.wait(updating_workers,
                                                         num_returns=1,
                                                         timeout=0.1)
                    upd = [ray.get(up) for up in updated]
                    idle_worker_ids.extend(upd)

                # if there are idle workers, use them to collect episodes
                # mark the newly busy workers as active
                while idle_worker_ids:
                    idle_worker_id = idle_worker_ids.pop()
                    worker = self._all_workers[idle_worker_id]
                    active_workers.append(worker.rollout.remote())

                # check which workers are done/not done collecting a sample
                # if any are done, send them to process the collected episode
                # if they are not, keep checking if they are done
                ready, not_ready = ray.wait(active_workers,
                                            num_returns=1,
                                            timeout=0.001)
                active_workers = not_ready
                for result in ready:
                    ready_worker_id, episode_batch = ray.get(result)
                    episodes[ready_worker_id].append(episode_batch)

                    if len(episodes[ready_worker_id]) < n_eps_per_worker:
                        idle_worker_ids.append(ready_worker_id)

                    pbar.update(1)

        ordered_episodes = list(
            itertools.chain(
                *[episodes[i] for i in range(self._worker_factory.n_workers)]))

        return EpisodeBatch.concatenate(*ordered_episodes)

    def shutdown_worker(self):
        """Shuts down the worker."""
        for worker in self._all_workers.values():
            worker.shutdown.remote()
        ray.shutdown()


class SamplerWorker:
    """Constructs a single sampler worker.

    Args:
        worker_id (int): The ID of this worker.
        env (Environment): Environment to sample form.
        agent_pkl (bytes): Pickled :class:`Policy` to sample with.
        worker_factory (WorkerFactory): Factory to construct this worker's
            behavior.

    """

    def __init__(self, worker_id, env, agent_pkl, worker_factory):
        # Must be called before pickle.loads below.
        self.inner_worker = worker_factory(worker_id)
        self.worker_id = worker_id
        self.inner_worker.update_env(env)
        self.inner_worker.update_agent(cloudpickle.loads(agent_pkl))

    def update(self, agent_update, env_update):
        """Update the agent and environment.

        Args:
            agent_update (object): Agent update.
            env_update (object): Environment update.

        Returns:
            int: The worker id.

        """
        self.inner_worker.update_agent(agent_update)
        self.inner_worker.update_env(env_update)
        return self.worker_id

    def rollout(self):
        """Sample one episode of the agent in the environment.

        Returns:
            tuple[int, EpisodeBatch]: Worker ID and batch of samples.

        """
        return (self.worker_id, self.inner_worker.rollout())

    def shutdown(self):
        """Shuts down the worker."""
        self.inner_worker.shutdown()
