"""This module implements a Vectorized Sampler used for OffPolicy Algorithms.

It diffs from OnPolicyVectorizedSampler in two parts:
 - The num of envs is defined by rollout_batch_size. In
 OnPolicyVectorizedSampler, the number of envs can be decided by batch_size
 and max_path_length. But OffPolicy algorithms usually samples transitions
 from replay buffer, which only has buffer_batch_size.
 - It needs to add transitions to replay buffer throughout the rollout.
"""

import itertools
import warnings

import cloudpickle
import numpy as np

from garage.experiment import deterministic
from garage.misc import tensor_utils
from garage.sampler.batch_sampler import BatchSampler
from garage.sampler.vec_env_executor import VecEnvExecutor


class OffPolicyVectorizedSampler(BatchSampler):
    """This class implements OffPolicyVectorizedSampler.

    Args:
        algo (garage.np.RLAlgorithm): Algorithm.
        env (garage.envs.GarageEnv): Environment.
        n_envs (int): Number of parallel environments managed by sampler.
        no_reset (bool): Reset environment between samples or not.

    """

    def __init__(self, algo, env, n_envs=None, no_reset=True):
        if n_envs is None:
            n_envs = int(algo.rollout_batch_size)
        super().__init__(algo, env)
        self._n_envs = n_envs
        self._no_reset = no_reset

        self._vec_env = None
        self._env_spec = self.env.spec

        self._last_obses = None
        self._last_uncounted_discount = [0] * n_envs
        self._last_running_length = [0] * n_envs
        self._last_success_count = [0] * n_envs

        warnings.warn(
            DeprecationWarning(
                'OffPolicyVectoriizedSampler is deprecated, and will be '
                'removed in the next release. Please use VecWorker and one of '
                'the new samplers which implement garage.sampler.Sampler, '
                'such as RaySampler.'))

    def start_worker(self):
        """Initialize the sampler."""
        n_envs = self._n_envs
        envs = [
            cloudpickle.loads(cloudpickle.dumps(self.env))
            for _ in range(n_envs)
        ]

        # Deterministically set environment seeds based on the global seed.
        seed0 = deterministic.get_seed()
        if seed0 is not None:
            for (i, e) in enumerate(envs):
                e.seed(seed0 + i)

        self._vec_env = VecEnvExecutor(
            envs=envs, max_path_length=self.algo.max_path_length)

    def shutdown_worker(self):
        """Terminate workers if necessary."""
        self._vec_env.close()

    # pylint: disable=too-many-branches, too-many-statements
    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        """Collect samples for the given iteration number.

        Args:
            itr(int): Iteration number.
            batch_size(int): Number of environment interactions in one batch.
            whole_paths(bool): Not effective. Only keep here to comply
                with base class.

        Raises:
            ValueError: If the algorithm doesn't have an exploration_policy
                field.

        Returns:
            list: A list of paths.

        """
        assert batch_size is not None

        paths = []
        if not self._no_reset or self._last_obses is None:
            obses = self._vec_env.reset()
        else:
            obses = self._last_obses
        completes = np.asarray([True] * self._vec_env.num_envs)
        running_paths = [None] * self._vec_env.num_envs
        n_samples = 0

        policy = self.algo.exploration_policy
        if policy is None:
            raise ValueError('OffPolicyVectoriizedSampler should only be used '
                             'with an exploration_policy.')
        while n_samples < batch_size:
            policy.reset(completes)
            obs_space = self.algo.env_spec.observation_space
            input_obses = obs_space.flatten_n(obses)

            actions, agent_infos = policy.get_actions(input_obses)

            next_obses, rewards, dones, env_infos = \
                self._vec_env.step(actions)
            completes = env_infos['vec_env_executor.complete']
            self._last_obses = next_obses
            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            n_samples += len(next_obses)

            if agent_infos is None:
                agent_infos = [dict() for _ in range(self._vec_env.num_envs)]
            if env_infos is None:
                env_infos = [dict() for _ in range(self._vec_env.num_envs)]

            for idx, reward, env_info, done, obs, next_obs, action in zip(
                    itertools.count(), rewards, env_infos, dones, obses,
                    next_obses, actions):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        rewards=[],
                        observations=[],
                        next_observations=[],
                        actions=[],
                        env_infos=[],
                        dones=[],
                        undiscounted_return=self._last_uncounted_discount[idx],
                        # running_length: Length of path up to now
                        # Note that running_length is not len(rewards)
                        # Because a path may not be complete in one batch
                        running_length=self._last_running_length[idx],
                        success_count=self._last_success_count[idx])

                running_paths[idx]['rewards'].append(reward)
                running_paths[idx]['observations'].append(obs)
                running_paths[idx]['next_observations'].append(next_obs)
                running_paths[idx]['actions'].append(action)
                running_paths[idx]['env_infos'].append(env_info)
                running_paths[idx]['dones'].append(done)
                running_paths[idx]['running_length'] += 1
                running_paths[idx]['undiscounted_return'] += reward
                running_paths[idx]['success_count'] += env_info.get(
                    'is_success') or 0

                self._last_uncounted_discount[idx] += reward
                self._last_success_count[idx] += env_info.get(
                    'is_success') or 0
                self._last_running_length[idx] += 1

                if done or n_samples >= batch_size:
                    paths.append(
                        dict(
                            rewards=np.asarray(running_paths[idx]['rewards']),
                            dones=np.asarray(running_paths[idx]['dones']),
                            env_infos=tensor_utils.stack_tensor_dict_list(
                                running_paths[idx]['env_infos']),
                            running_length=running_paths[idx]
                            ['running_length'],
                            undiscounted_return=running_paths[idx]
                            ['undiscounted_return'],
                            success_count=running_paths[idx]['success_count']))

                    act_space = self._env_spec.action_space
                    path_dict = {}

                    path_dict['observations'] = obs_space.flatten_n(
                        running_paths[idx]['observations'])
                    path_dict['next_observations'] = obs_space.flatten_n(
                        running_paths[idx]['next_observations'])
                    path_dict['rewards'] = np.asarray(
                        running_paths[idx]['rewards']).reshape(-1, 1)
                    path_dict['terminals'] = np.asarray(
                        running_paths[idx]['dones']).reshape(-1, 1)
                    path_dict['actions'] = act_space.flatten_n(
                        running_paths[idx]['actions'])

                    self.algo.replay_buffer.add_path(path_dict)
                    running_paths[idx] = None

                    if done:
                        self._last_running_length[idx] = 0
                        self._last_success_count[idx] = 0
                        self._last_uncounted_discount[idx] = 0
            obses = next_obses
        return paths
