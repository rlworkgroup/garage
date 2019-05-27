import itertools
import pickle

from dowel import logger, tabular
import numpy as np

from garage.misc import tensor_utils
from garage.misc.overrides import overrides
from garage.misc.prog_bar_counter import ProgBarCounter
from garage.sampler.utils import truncate_paths
from garage.tf.envs import VecEnvExecutor
from garage.tf.samplers.batch_sampler import BatchSampler


class OnPolicyVectorizedSampler(BatchSampler):
    def __init__(self, algo, env, n_envs=1):
        super(OnPolicyVectorizedSampler, self).__init__(algo, env, n_envs)
        self.n_envs = n_envs

    @overrides
    def start_worker(self):
        n_envs = self.n_envs
        envs = [pickle.loads(pickle.dumps(self.env)) for _ in range(n_envs)]
        self.vec_env = VecEnvExecutor(
            envs=envs, max_path_length=self.algo.max_path_length)
        self.env_spec = self.env.spec

    @overrides
    def shutdown_worker(self):
        self.vec_env.close()

    @overrides
    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
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

        import time
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
                    paths.append(
                        dict(
                            observations=self.env_spec.observation_space.
                            flatten_n(running_paths[idx]['observations']),
                            actions=self.env_spec.action_space.flatten_n(
                                running_paths[idx]['actions']),
                            rewards=tensor_utils.stack_tensor_list(
                                running_paths[idx]['rewards']),
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

        if whole_paths:
            return paths
        else:
            paths_truncated = truncate_paths(paths, batch_size)
            return paths_truncated
