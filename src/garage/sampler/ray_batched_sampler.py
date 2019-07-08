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

from garage.misc import tensor_utils
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.sampler.base import BaseSampler


class RaySampler(BaseSampler):
    """Collects Policy Rollouts in a data parallel fashion.

    Args:
        - algo: A garage algo object
        - env: A gym/akro env object
        - should_render(bool): should the sampler render the trajectories
        - sampler_worker_cls: If none, uses the default SamplerWorker
            class

    """

    def __init__(self,
                 algo,
                 env,
                 should_render=False,
                 num_processors=None,
                 sampler_worker_cls=None):
        self.SamplerWorker = ray.remote(SamplerWorker if sampler_worker_cls is
                                        None else sampler_worker_cls)

        self.env = env
        self.algo = algo
        self.max_path_length = self.algo.max_path_length
        self.should_render = should_render
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        self.num_workers = num_processors if num_processors \
            else psutil.cpu_count(logical=False)
        self.all_workers = defaultdict(None)
        self.active_workers = []
        self.active_worker_ids = []

    def start_worker(self):
        """Initialize a new ray worker."""
        env_pkl = pickle.dumps(self.env)
        agent_pkl = pickle.dumps(self.algo.policy)
        env_pkl_id, agent_pkl_id = ray.put(env_pkl), ray.put(agent_pkl)
        for worker_id in range(self.num_workers):
            self.all_workers[worker_id] = (self.SamplerWorker.remote(
                worker_id, env_pkl_id, agent_pkl_id, self.max_path_length,
                self.should_render))
        self.idle_worker_ids = list(range(self.num_workers))

    def obtain_samples(self, itr, num_samples):
        """Sample the policy for new trajectories.

        Args:
            - itr(int): iteration number
            - num_samples(int):number of steps the the sampler should collect
        """
        pbar = ProgBarCounter(num_samples)
        completed_samples = 0
        traj = []
        updating_workers = []
        self.idle_worker_ids = list(range(self.num_workers))

        curr_policy_params = self.algo.policy.get_param_values()
        params_id = ray.put(curr_policy_params)
        while self.idle_worker_ids:
            worker_id = self.idle_worker_ids.pop()
            worker = self.all_workers[worker_id]
            updating_workers.append(worker.set_agent.remote(params_id))

        while completed_samples < num_samples:
            updated, updating_workers = ray.wait(
                updating_workers, num_returns=1, timeout=0.1)
            upd = [ray.get(up) for up in updated]
            self.idle_worker_ids.extend(upd)
            while self.idle_worker_ids:
                idle_worker_id = self.idle_worker_ids.pop()
                self.active_worker_ids.append(idle_worker_id)
                worker = self.all_workers[idle_worker_id]
                self.active_workers.append(worker.rollout.remote())

            ready, not_ready = ray.wait(
                self.active_workers, num_returns=1, timeout=0.001)
            self.active_workers = not_ready
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
        trajectory = ray.get(result)
        ready_worker_id = trajectory[0]
        self.active_worker_ids.remove(ready_worker_id)
        self.idle_worker_ids.append(ready_worker_id)
        trajectory = dict(
            observations=self.algo.env_spec.observation_space.flatten_n(
                trajectory[1]),
            actions=self.algo.env_spec.action_space.flatten_n(trajectory[2]),
            rewards=tensor_utils.stack_tensor_list(trajectory[3]),
            agent_infos=trajectory[4],
            env_infos=trajectory[5])
        num_returned_samples = len(trajectory['observations'])
        return trajectory, num_returned_samples


class SamplerWorker:
    """Constructs a single sampler worker.

    The worker can have its parameters updated, and sampler its policy for
    trajectories or rollouts.

    Args:
        - worker_id(int): the id of the sampler_worker
        - env: gym or akro env object
        - max_path_length(int): max trajectory length
        - should_render(bool): if true, renders trajectories after
            sampling them

    """

    def __init__(self,
                 worker_id,
                 env,
                 agent,
                 max_path_length,
                 should_render=False):
        self.worker_id = worker_id
        self.env = pickle.loads(env)
        self.agent = pickle.loads(agent)
        self.max_path_length = max_path_length
        self.should_render = should_render
        self.agent_updates = 0

    def set_agent(self, flattened_params):
        """Set the agent params.

        Args:
            - flattened_params(): model parameters in numpy format
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
        """
        observations = []
        actions = []
        rewards = []
        agent_infos = defaultdict(list)
        env_infos = defaultdict(list)
        o = self.env.reset()
        self.agent.reset()
        next_o = None
        path_length = 0
        while path_length < self.max_path_length:
            a, agent_info = self.agent.get_action(o)
            next_o, r, d, env_info = self.env.step(a)
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
