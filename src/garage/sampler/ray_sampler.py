"""This is an implementation of an on policy batch sampler.

Uses a data parallel design.
Included is a sampler that deploys sampler workers.

The sampler workers must implement some type of set agent parameters
function, and a rollout function
"""
from collections import defaultdict
import pickle

import psutil
import ray

from garage.misc import tensor_utils
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.sampler.base import Sampler


class RaySampler(Sampler):
    """Collects Policy Rollouts in a data parallel fashion.

    Args:
        - algo: A garage algo object
        - env: A gym/akro env object
        - sampler_worker_cls: If none, uses the default SamplerWorker
            class

    """

    def __init__(self,
                 config,
                 agent,
                 env,
                 num_processors=None,
                 sampler_worker_cls=None):
        self._sampler_worker = ray.remote(SamplerWorker if sampler_worker_cls
                                          is None else sampler_worker_cls)
        self._config = config
        self._agent = agent
        self._env = env
        if not ray.is_initialized():
            ray.init(log_to_driver=False)
        self._num_workers = (num_processors if num_processors else
                             psutil.cpu_count(logical=False))
        self._all_workers = defaultdict(None)
        self._workers_started = False
        self.start_worker()
        # TODO: Clean up these fields, since they can just be local vairables.
        self._active_workers = []
        self._active_worker_ids = []
        self._idle_worker_ids = list(range(self._num_workers))

    @classmethod
    def construct(cls, config, agent, env):
        """Construct this sampler from a config."""
        return cls(config, agent, env)

    def start_worker(self):
        """Initialize a new ray worker."""
        if self._workers_started:
            return
        else:
            self._workers_started = True
        # Pickle the environment once, instead of once per worker.
        env_pkl = pickle.dumps(self._env)
        # We need to pickle the agent so that we can e.g. set up the TF Session
        # in the worker *before* unpicling it.
        agent_pkl = pickle.dumps(self._agent)
        for worker_id in range(self._num_workers):
            self._all_workers[worker_id] = self._sampler_worker.remote(
                worker_id, env_pkl, agent_pkl, self._config)

    def obtain_samples(self, itr, num_samples, agent_update, env_update=None):
        """Sample the policy for new trajectories.

        Args:
            - itr(int): iteration number
            - num_samples(int):number of steps the the sampler should collect
        """
        self._active_workers = []
        self._active_worker_ids = []
        pbar = ProgBarCounter(num_samples)
        completed_samples = 0
        traj = []
        updating_workers = []

        # update the policy params of each worker before sampling
        # for the current iteration
        self._idle_worker_ids = list(range(self._num_workers))
        params_id = ray.put(agent_update)
        env_id = ray.put(env_update)
        while self._idle_worker_ids:
            worker_id = self._idle_worker_ids.pop()
            worker = self._all_workers[worker_id]
            updating_workers.append(worker.update.remote(params_id, env_id))

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
                self._active_workers.append(worker.rollout.remote())

            # check which workers are done/not done collecting a sample
            # if any are done, send them to process the collected trajectory
            # if they are not, keep checking if they are done
            ready, not_ready = ray.wait(self._active_workers,
                                        num_returns=1,
                                        timeout=0.001)
            self._active_workers = not_ready
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
            - result: ray object id of ready to be collected trajectory.
        """
        trajectory = ray.get(result)
        ready_worker_id = trajectory[0]
        self._active_worker_ids.remove(ready_worker_id)
        self._idle_worker_ids.append(ready_worker_id)
        trajectory = dict(
            observations=self._env.observation_space.flatten_n(trajectory[1]),
            actions=self._env.action_space.flatten_n(trajectory[2]),
            rewards=tensor_utils.stack_tensor_list(trajectory[3]),
            agent_infos=trajectory[4],
            env_infos=trajectory[5])
        num_returned_samples = len(trajectory['observations'])
        return trajectory, num_returned_samples


class SamplerWorker:
    """Constructs a single sampler worker.

    Args:
        - worker_id(int): The id of the sampler_worker
        - env_pkl: The pickled gym env
        - agent_pkl: The pickled agent
        - config(SamplerConfig): The sampler configuration

    """

    def __init__(self, worker_id, env_pkl, agent_pkl, config):
        config.worker_init_fn(config, worker_id)
        self.worker_id = worker_id
        self.env = pickle.loads(env_pkl)
        self.agent = pickle.loads(agent_pkl)
        self.config = config

    def update(self, agent_update, env_update):
        """Update the agent and environment."""
        self.agent = self.config.agent_update_fn(self.config, self.worker_id,
                                                 self.agent, agent_update)
        self.env = self.config.env_update_fn(self.config, self.worker_id,
                                             self.env, env_update)
        return self.worker_id

    def rollout(self):
        """Compute one rollout of the agent in the environment."""
        return self.config.rollout_fn(self.config, self.worker_id, self.agent,
                                      self.env, self.config.max_path_length)
