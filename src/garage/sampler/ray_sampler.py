"""This is an implementation of an on policy batch sampler.

Uses a data parallel design.
Included is a sampler that deploys sampler workers.

The sampler workers must implement some type of set agent parameters
function, and a rollout function
"""
from collections import defaultdict
import pickle

import numpy as np
import psutil
import ray

from garage.experiment import deterministic
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.sampler.base import BaseSampler


class RaySampler(BaseSampler):
    """Collects Policy Rollouts in a data parallel fashion.

    Args:
        algo (garage.np.algo.RLAlgorithm): A garage algo object
        env (gym.Env): A gym/akro env object
        seed (int): Random seed.
        should_render (bool): If the sampler render the trajectories.
        num_processors (int): Number of processors to be used.
        sampler_worker_cls (garage.sampler.ray_sampler.SamplerWorker):
            If none, uses the default SamplerWorker class

    """

    def __init__(self,
                 algo,
                 env,
                 seed,
                 should_render=False,
                 num_processors=None,
                 sampler_worker_cls=None):
        super().__init__(algo, env)
        self._sampler_worker = ray.remote(SamplerWorker if sampler_worker_cls
                                          is None else sampler_worker_cls)
        self._seed = seed
        deterministic.set_seed(self._seed)
        self._max_path_length = self.algo.max_path_length
        self._should_render = should_render
        if not ray.is_initialized():
            ray.init(log_to_driver=False)
        self._num_workers = (num_processors if num_processors else
                             psutil.cpu_count(logical=False))
        self._all_workers = defaultdict(None)
        self._idle_worker_ids = list(range(self._num_workers))
        self._active_worker_ids = []

    def start_worker(self):
        """Initialize a new ray worker."""
        # Pickle the environment once, instead of once per worker.
        env_pkl = pickle.dumps(self.env)
        # We need to pickle the agent so that we can e.g. set up the TF Session
        # in the worker *before* unpicling it.
        agent_pkl = pickle.dumps(self.algo.policy)
        for worker_id in range(self._num_workers):
            self._all_workers[worker_id] = self._sampler_worker.remote(
                worker_id, env_pkl, agent_pkl, self._seed,
                self._max_path_length, self._should_render)

    # pylint: disable=arguments-differ
    def obtain_samples(self, itr, num_samples):
        """Sample the policy for new trajectories.

        Args:
            itr (int): Iteration number.
            num_samples (int): Number of steps the the sampler should collect.

        Returns:
            list[dict]: Sample paths, each path with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)

        """
        _active_workers = []
        self._active_worker_ids = []
        pbar = ProgBarCounter(num_samples)
        completed_samples = 0
        traj = []
        updating_workers = []

        # update the policy params of each worker before sampling
        # for the current iteration
        curr_policy_params = self.algo.policy.get_param_values()
        params_id = ray.put(curr_policy_params)
        while self._idle_worker_ids:
            worker_id = self._idle_worker_ids.pop()
            worker = self._all_workers[worker_id]
            updating_workers.append(worker.set_agent.remote(params_id))

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
                self._idle_worker_ids.extend(upd)

            # if there are idle workers, use them to collect trajectories
            # mark the newly busy workers as active
            while self._idle_worker_ids:
                idle_worker_id = self._idle_worker_ids.pop()
                self._active_worker_ids.append(idle_worker_id)
                worker = self._all_workers[idle_worker_id]
                _active_workers.append(worker.rollout.remote())

            # check which workers are done/not done collecting a sample
            # if any are done, send them to process the collected trajectory
            # if they are not, keep checking if they are done
            ready, not_ready = ray.wait(_active_workers,
                                        num_returns=1,
                                        timeout=0.001)
            _active_workers = not_ready
            for result in ready:
                trajectory, num_returned_samples = self._process_trajectory(
                    result)
                completed_samples += num_returned_samples
                pbar.inc(num_returned_samples)
                traj.append(trajectory)
        pbar.stop()
        return traj

    def shutdown_worker(self):
        """Shuts down the worker."""
        ray.shutdown()

    def _process_trajectory(self, result):
        """Collect trajectory from ray object store.

        Converts that trajectory to garage friendly format.

        Args:
            result (obj): Ray object id of ready to be collected trajectory.

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
        self._active_worker_ids.remove(ready_worker_id)
        self._idle_worker_ids.append(ready_worker_id)
        trajectory = dict(observations=np.asarray(trajectory[1]),
                          actions=np.asarray(trajectory[2]),
                          rewards=np.asarray(trajectory[3]),
                          agent_infos=trajectory[4],
                          env_infos=trajectory[5])
        num_returned_samples = len(trajectory['observations'])
        return trajectory, num_returned_samples


class SamplerWorker:
    """Constructs a single sampler worker.

    The worker can have its parameters updated, and sampler its policy for
    trajectories or rollouts.

    Args:
        worker_id (int): the id of the sampler_worker
        env_pkl (bytes): A pickled gym or akro env object
        agent_pkl (bytes): A pickled agent
        seed (int): Random seed.
        max_path_length (int): max trajectory length
        should_render (bool): if true, renders trajectories after
            sampling them

    """

    def __init__(self,
                 worker_id,
                 env_pkl,
                 agent_pkl,
                 seed,
                 max_path_length,
                 should_render=False):
        self.worker_id = worker_id
        self._env = pickle.loads(env_pkl)
        self.agent = pickle.loads(agent_pkl)
        self._seed = seed + self.worker_id
        deterministic.set_seed(self._seed)
        self._max_path_length = max_path_length
        self._should_render = should_render
        self.agent_updates = 0

    def set_agent(self, flattened_params):
        """Set the agent params.

        Args:
            flattened_params (list[np.ndarray]): model parameters

        Returns:
            int: Worker id of this sampler worker.

        """
        self.agent.set_param_values(flattened_params)
        self.agent_updates += 1
        return self.worker_id

    def rollout(self):
        """Sample a single rollout from the agent/policy.

        The following value for the following keys will be a 2D array,
        with the first dimension corresponding to the time dimension.

        - observations
        - actions
        - rewards
        - next_observations
        - terminals
        The next two elements will be lists of dictionaries, with
        the index into the list being the index into the time
        - agent_infos
        - env_infos

        Returns:
            int: ID of this work
            numpy.ndarray: observations
            numpy.ndarray: actions
            numpy.ndarray: rewards
            dict[list]: agent info
            dict[list]: environment info

        """
        observations = []
        actions = []
        rewards = []
        agent_infos = defaultdict(list)
        env_infos = defaultdict(list)
        o = self._env.reset()
        self.agent.reset()
        next_o = None
        path_length = 0
        while path_length < self._max_path_length:
            a, agent_info = self.agent.get_action(o)
            next_o, r, d, env_info = self._env.step(a)
            observations.append(o)
            rewards.append(r)
            actions.append(a)
            for k, v in agent_info.items():
                agent_infos[k].append(v)
            for k, v in env_info.items():
                env_infos[k].append(v)
            path_length += 1
            if d:
                break
            o = next_o
        for k, v in agent_infos.items():
            agent_infos[k] = np.asarray(v)
        for k, v in env_infos.items():
            env_infos[k] = np.asarray(v)
        return self.worker_id,\
            np.array(observations),\
            np.array(actions),\
            np.array(rewards),\
            dict(agent_infos),\
            dict(env_infos)
