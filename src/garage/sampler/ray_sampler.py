"""This is an implementation of an on policy batch sampler.

Uses a data parallel design.
Included is a sampler that deploys sampler workers.
The sampler workers must implement some type of set agent parameters
function, and a rollout function.

"""
from collections import defaultdict
import pickle

import numpy as np
import psutil
import ray

from garage.misc.prog_bar_counter import ProgBarCounter
from garage.sampler.sampler import Sampler


class RaySampler(Sampler):
    """Collects Policy Rollouts in a data parallel fashion.

    Args:
        worker_factory(garage.sampler.WorkerFactory): Used for worker behavior.
        agents(list[garage.Policy]): Agents to distribute across workers.
        envs(list[gym.Env]): Environments to distribute across workers.
        num_processors(int): Number of workers processes to spawn. If none,
            defaults to number of physical CPU cores.
        sampler_worker_cls(type): If None, uses the default SamplerWorker
            class.

    """

    def __init__(self,
                 worker_factory,
                 agents,
                 envs,
                 num_processors=None,
                 sampler_worker_cls=None):
        # pylint: disable=super-init-not-called
        self._sampler_worker = ray.remote(SamplerWorker if sampler_worker_cls
                                          is None else sampler_worker_cls)
        self._worker_factory = worker_factory
        self._agents = agents
        self._envs = self._worker_factory.prepare_worker_messages(envs)
        if not ray.is_initialized():
            ray.init(log_to_driver=False)
        self._num_workers = (num_processors if num_processors else
                             psutil.cpu_count(logical=False))
        self._all_workers = defaultdict(None)
        self._workers_started = False
        self.start_worker()

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

    def start_worker(self):
        """Initialize a new ray worker."""
        if self._workers_started:
            return
        self._workers_started = True
        # We need to pickle the agent so that we can e.g. set up the TF.Session
        # in the worker *before* unpicling it.
        agent_pkls = self._worker_factory.prepare_worker_messages(
            self._agents, pickle.dumps)
        for worker_id in range(self._num_workers):
            self._all_workers[worker_id] = self._sampler_worker.remote(
                worker_id, self._envs[worker_id], agent_pkls[worker_id],
                self._worker_factory)

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Sample the policy for new trajectories.

        Args:
            itr(int): Iteration number.
            num_samples(int): Number of steps the the sampler should collect.
            agent_update(object): Value which will be passed into the
                `agent_update_fn` before doing rollouts. If a list is passed
                in, it must have length exactly `factory.n_workers`, and will
                be spread across the workers.
            env_update(object): Value which will be passed into the
                `env_update_fn` before doing rollouts. If a list is passed in,
                it must have length exactly `factory.n_workers`, and will be
                spread across the workers.

        Returns:
            list[dict]: Sample paths, each path with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)

        """
        active_workers = []
        active_worker_ids = []
        idle_worker_ids = list(range(self._num_workers))
        pbar = ProgBarCounter(num_samples)
        completed_samples = 0
        traj = []
        updating_workers = []

        # update the policy params of each worker before sampling
        # for the current iteration
        idle_worker_ids = list(range(self._num_workers))
        param_ids = self._worker_factory.prepare_worker_messages(
            agent_update, ray.put)
        env_ids = self._worker_factory.prepare_worker_messages(
            env_update, ray.put)
        while idle_worker_ids:
            worker_id = idle_worker_ids.pop()
            worker = self._all_workers[worker_id]
            updating_workers.append(
                worker.update.remote(param_ids[worker_id], env_ids[worker_id]))

        while completed_samples < num_samples:
            # if there are workers still being updated, check
            # which ones are still updating and take the workers that
            # are done updating, and start collecting trajectories on
            # those workers.
            if updating_workers:
                updated, updating_workers = ray.wait(updating_workers,
                                                     num_returns=1,
                                                     timeout=0.1)
                upd = [ray.get(up) for up in updated]
                idle_worker_ids.extend(upd)

            # if there are idle workers, use them to collect trajectories
            # mark the newly busy workers as active
            while idle_worker_ids:
                idle_worker_id = idle_worker_ids.pop()
                active_worker_ids.append(idle_worker_id)
                worker = self._all_workers[idle_worker_id]
                active_workers.append(worker.rollout.remote())

            # check which workers are done/not done collecting a sample
            # if any are done, send them to process the collected trajectory
            # if they are not, keep checking if they are done
            ready, not_ready = ray.wait(active_workers,
                                        num_returns=1,
                                        timeout=0.001)
            active_workers = not_ready
            for result in ready:
                trajectory, num_returned_samples = _process_trajectory(
                    result, active_worker_ids, idle_worker_ids)
                completed_samples += num_returned_samples
                pbar.inc(num_returned_samples)
                traj.append(trajectory)
        pbar.stop()
        return traj

    def shutdown_worker(self):
        """Shuts down the worker."""
        ray.shutdown()


def _process_trajectory(result, active_worker_ids, idle_worker_ids):
    """Collect trajectory from ray object store.

    Converts that trajectory to garage friendly format.

    Args:
        result(object): Ray object id of ready to be collected trajectory.
        active_worker_ids(list[int]): Active worker IDs. Workers with
            completed trajectories will be removed.
        idle_worker_ids(list[int]): Idle worker IDs. Workers with completed
            trajectories will be added.

    Returns:
        dict: One trajectory, with keys
            * observations: (numpy.ndarray)
            * actions: (numpy.ndarray)
            * rewards: (numpy.ndarray)
            * agent_infos: (dict)
            * env_infos: (dict)
        int: Number of returned samples in the trajectory

    """
    trajectory = ray.get(result)
    ready_worker_id = trajectory[0]
    active_worker_ids.remove(ready_worker_id)
    idle_worker_ids.append(ready_worker_id)
    trajectory = dict(observations=np.asarray(trajectory[1]),
                      actions=np.asarray(trajectory[2]),
                      rewards=np.asarray(trajectory[3]),
                      agent_infos=trajectory[4],
                      env_infos=trajectory[5])
    num_returned_samples = len(trajectory['observations'])
    return trajectory, num_returned_samples


class SamplerWorker:
    """Constructs a single sampler worker.

    Args:
        worker_id(int): The id of the sampler_worker
        env(gym.Env): The gym env
        agent_pkl(bytes): The pickled agent
        worker_factory(WorkerFactory): Factory to construct this worker's
            behavior.

    """

    def __init__(self, worker_id, env, agent_pkl, worker_factory):
        self.inner_worker = worker_factory(worker_id)
        self.worker_id = worker_id
        self.inner_worker.env = env
        self.inner_worker.agent = pickle.loads(agent_pkl)

    def update(self, agent_update, env_update):
        """Update the agent and environment.

        Args:
            agent_update(object): Agent update.
            env_update(object): Environment update.

        Returns:
            int: The worker id.

        """
        self.inner_worker.update_agent(agent_update)
        self.inner_worker.update_env(env_update)
        return self.worker_id

    def rollout(self):
        """Compute one rollout of the agent in the environment.

        Returns:
            tuple: Worker id and return values of `Worker.rollout`.

        """
        return (self.worker_id, ) + self.inner_worker.rollout()
