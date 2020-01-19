"""RL2 Sampler which uses VecEnvExecutor to run multiple environments."""
import copy
import itertools
import time

from dowel import logger, tabular
import numpy as np

from garage.experiment import deterministic
from garage.misc import tensor_utils
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.sampler.base import BaseSampler
from garage.sampler.stateful_pool import singleton_pool
from garage.sampler.utils import truncate_paths
from garage.sampler.vec_env_executor import VecEnvExecutor


class RL2Sampler(BaseSampler):
    """Sampler which uses VecEnvExecutor to run multiple environments.

    This sampler is for RL^2. See https://arxiv.org/pdf/1611.02779.pdf.

    In RL^2, there are n environments/tasks and paths in each of them
    will be concatenated at some point and fed to the policy.
    This sampler uses an OrderedDict, instead of a List, to keep track
    of the paths for each environment/task.

    Args:
        algo (garage.np.algos.RLAlgorithm): An algorithm instance.
        env (garage.envs.GarageEnv): Environement to sample from.
        meta_batch_size (int): Meta batch size for sampling. If it is
            larger than n_envs, it must be a multiple of n_envs so it can be
            evenly distributed among environments.
        n_envs (int): Number of environment instances for sampling. It it is
            larger than meta_batch_size, it must be a multiple of
            meta_batch_size so batch can be evenly distributed among
            environments.

    Raises:
        ValueError: If meta_batch_size > n_envs and meta_batch_size is not
            a multiple of n_envs, or if n_envs > meta_batch_size and n_envs
            is not a multiple of meta_batch_size.

    """

    def __init__(self, algo, env, meta_batch_size, n_envs=None):
        super().__init__(algo, env)
        if n_envs is None:
            n_envs = singleton_pool.n_parallel * 4

        self._n_envs = n_envs
        self._meta_batch_size = meta_batch_size
        self._vec_env = None
        self._envs_per_worker = None
        self._vec_envs_indices = None

        if self._meta_batch_size > self._n_envs:
            if self._meta_batch_size % self._n_envs != 0:
                raise ValueError(
                    'meta_batch_size must be a multiple of n_envs')
            self._envs_per_worker = 1
            self._vec_envs_indices = np.split(np.arange(self._meta_batch_size),
                                              self._n_envs)
        if self._n_envs >= self._meta_batch_size:
            if self._n_envs % self._meta_batch_size != 0:
                raise ValueError(
                    'n_envs must be a multiple of meta_batch_size')
            self._envs_per_worker = self._n_envs // self._meta_batch_size
            self._vec_envs_indices = [np.arange(self._meta_batch_size)]

    def start_worker(self):
        """This function is deprecated."""

    def shutdown_worker(self):
        """Shutdown workers."""
        self._vec_env.close()

    def _setup_worker(self, env_indices, tasks):
        """Setup workers.

        Args:
            env_indices (List[Int]): Indices of environments to be assigned
                to workers for sampling.
            tasks (List[dict]): List of tasks to assign.

        """
        if self._vec_env is not None:
            self._vec_env.close()

        vec_envs = []
        for env_ind in env_indices:
            for _ in range(self._envs_per_worker):
                vec_env = copy.deepcopy(self.env)
                vec_env.set_task(tasks[env_ind])
                vec_envs.append(vec_env)
        seed0 = deterministic.get_seed()
        if seed0 is not None:
            for (i, e) in enumerate(vec_envs):
                e.seed(seed0 + i)

        self._vec_env = VecEnvExecutor(
            envs=vec_envs, max_path_length=self.algo.max_path_length)

    # pylint: disable=too-many-statements
    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        """Sample the policy for new trajectories.

        If batch size is not specified, episode per task by default is 1 so
        batch size will be meta_batch_size * max_path_length.

        When number of workers are less than meta batch size, sampling will
        be performed for each of self._vec_envs_indices in series. The
        i-th value of self._vec_envs_indices represents the indices of the
        environments/tasks to be sampled for the i-th iteration.

        Args:
            itr (int): Iteration number.
            batch_size (int): Number of samples to be collected. If None,
                it will be default [algo.max_path_length * n_envs].
            whole_paths (bool): Whether return all the paths or not. True
                by default. It's possible for the paths to have total actual
                sample size larger than batch_size, and will be truncated if
                this flag is true.

        Returns:
            OrderedDict: Sample paths. Key represents the index of the
                environment/task and value represents all the paths sampled
                from that particular environment/task.


        Note:
            Each path is a dictionary, with keys and values as following:
                * observations: numpy.ndarray with shape :math:`[N, S^*]`
                * actions: numpy.ndarray with shape :math:`[N, S^*]`
                * rewards: numpy.ndarray with shape :math:`[N, S^*]`
                * dones: numpy.ndarray with shape :math:`[N, S^*]`
                * env_infos: A dictionary with each key representing one
                  environment info, value being a numpy.ndarray with shape
                  :math:`[N, S^*]`. One example is "ale.lives" for atari
                  environments.
                * agent_infos: A dictionary with each key representing one
                  agent info, value being a numpy.ndarray with shape
                  :math:`[N, S^*]`. One example is "prev_action", which is used
                  for recurrent policy as previous action input, merged with
                  the observation input as the state input.

        """
        logger.log('Obtaining samples for iteration %d...' % itr)

        if batch_size is None:
            batch_size = self.algo.max_path_length * self._meta_batch_size

        paths = []

        tasks = self.env.sample_tasks(self._meta_batch_size)

        # Start main loop
        batch_size_per_loop = batch_size // len(self._vec_envs_indices)
        for vec_envs_indices in self._vec_envs_indices:
            self._setup_worker(vec_envs_indices, tasks)

            n_samples = 0
            obses = self._vec_env.reset()
            dones = np.asarray([True] * self._vec_env.num_envs)
            running_paths = [None] * self._vec_env.num_envs

            pbar = ProgBarCounter(batch_size)
            policy_time = 0
            env_time = 0
            process_time = 0

            policy = self.algo.policy
            # Only reset policies at the beginning of a meta batch
            policy.reset(dones)

            while n_samples < batch_size_per_loop:
                t = time.time()

                actions, agent_infos = policy.get_actions(obses)

                policy_time += time.time() - t
                t = time.time()
                next_obses, rewards, dones, env_infos = self._vec_env.step(
                    actions)
                env_time += time.time() - t
                t = time.time()

                agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
                env_infos = tensor_utils.split_tensor_dict_list(env_infos)
                if env_infos is None:
                    env_infos = [dict() for _ in range(self._vec_env.num_envs)]
                if agent_infos is None:
                    agent_infos = [
                        dict() for _ in range(self._vec_env.num_envs)
                    ]
                for idx, observation, action, reward, env_info, agent_info, done in zip(  # noqa: E501
                        itertools.count(), obses, actions, rewards, env_infos,
                        agent_infos, dones):
                    if running_paths[idx] is None:
                        running_paths[idx] = dict(
                            observations=[],
                            actions=[],
                            rewards=[],
                            dones=[],
                            env_infos=[],
                            agent_infos=[],
                        )
                    running_paths[idx]['observations'].append(observation)
                    running_paths[idx]['actions'].append(action)
                    running_paths[idx]['rewards'].append(reward)
                    running_paths[idx]['dones'].append(done)
                    running_paths[idx]['env_infos'].append(env_info)
                    running_paths[idx]['agent_infos'].append(agent_info)
                    if done:
                        obs = np.asarray(running_paths[idx]['observations'])
                        actions = np.asarray(running_paths[idx]['actions'])
                        paths.append(
                            dict(observations=obs,
                                 actions=actions,
                                 rewards=np.asarray(
                                     running_paths[idx]['rewards']),
                                 dones=np.asarray(running_paths[idx]['dones']),
                                 env_infos=tensor_utils.stack_tensor_dict_list(
                                     running_paths[idx]['env_infos']),
                                 agent_infos=tensor_utils.
                                 stack_tensor_dict_list(
                                     running_paths[idx]['agent_infos']),
                                 batch_idx=idx))
                        n_samples += len(running_paths[idx]['rewards'])
                        running_paths[idx] = None

                process_time += time.time() - t
                pbar.inc(len(obses))
                obses = next_obses

        pbar.stop()

        tabular.record('PolicyExecTime', policy_time)
        tabular.record('EnvExecTime', env_time)
        tabular.record('ProcessExecTime', process_time)

        return paths if whole_paths else truncate_paths(paths, batch_size)
