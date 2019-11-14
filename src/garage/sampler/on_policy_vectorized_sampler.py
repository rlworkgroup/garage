"""BatchSampler which uses VecEnvExecutor to run multiple environments."""
import itertools
import pickle
import time

from dowel import logger, tabular
import numpy as np

from garage.experiment import deterministic
from garage.misc import tensor_utils
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.sampler.batch_sampler import BatchSampler
from garage.sampler.stateful_pool import singleton_pool
from garage.sampler.utils import truncate_paths
from garage.sampler.vec_env_executor import VecEnvExecutor


class OnPolicyVectorizedSampler(BatchSampler):
    """BatchSampler which uses VecEnvExecutor to run multiple environments.

    Args:
        algo (garage.np.algo.RLAlgorithm): A garage algo object
        env (gym.Env): A gym/akro env object
        n_envs (int): Number of parallel environments used for sampling.

    """

    def __init__(self, algo, env, n_envs=None):
        if n_envs is None:
            n_envs = singleton_pool.n_parallel * 4
        super().__init__(algo, env)
        self.n_envs = n_envs
        self.env_spec = self.env.spec
        self.vec_env = None

    def start_worker(self):
        """Start workers."""
        n_envs = self.n_envs
        envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_envs)]

        # Deterministically set environment seeds based on the global seed.
        for (i, e) in enumerate(envs):
            e.seed(deterministic.get_seed() + i)

        self.vec_env = VecEnvExecutor(
            envs=envs, max_path_length=self.algo.max_path_length)

    def shutdown_worker(self):
        """Shutdown workers."""
        self.vec_env.close()

    # pylint: disable=too-many-statements
    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        """Sample the policy for new trajectories.

        Args:
            itr (int): Iteration number.
            batch_size (int): Number of samples to be collected. If None,
                it will be default [algo.max_path_length * n_envs].
            whole_paths (bool): Whether return all the paths or not. True
                by default. It's possible for the paths to have total actual
                sample size larger than batch_size, and will be truncated if
                this flag is true.

        Returns:
            list[dict]: Sample paths, each path with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)

        """
        logger.log('Obtaining samples for iteration %d...' % itr)

        if not batch_size:
            batch_size = self.algo.max_path_length * self.n_envs

        paths = []
        n_samples = 0
        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy

        while n_samples < batch_size:
            t = time.time()
            policy.reset(dones)

            actions, agent_infos = policy.get_actions(obses)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t
            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            for idx, observation, action, reward, env_info, agent_info, done in zip(  # noqa: E501
                    itertools.count(), obses, actions, rewards, env_infos,
                    agent_infos, dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]['observations'].append(observation)
                running_paths[idx]['actions'].append(action)
                running_paths[idx]['rewards'].append(reward)
                running_paths[idx]['env_infos'].append(env_info)
                running_paths[idx]['agent_infos'].append(agent_info)
                if done:
                    obs = np.asarray(running_paths[idx]['observations'])
                    actions = np.asarray(running_paths[idx]['actions'])
                    paths.append(
                        dict(observations=obs,
                             actions=actions,
                             rewards=np.asarray(running_paths[idx]['rewards']),
                             env_infos=tensor_utils.stack_tensor_dict_list(
                                 running_paths[idx]['env_infos']),
                             agent_infos=tensor_utils.stack_tensor_dict_list(
                                 running_paths[idx]['agent_infos'])))
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
