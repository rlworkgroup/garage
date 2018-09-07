import itertools
import pickle

import numpy as np

from garage.misc import tensor_utils
from garage.misc.overrides import overrides
from garage.replay_buffer.base import Buffer
from garage.tf.envs import VecEnvExecutor
from garage.tf.samplers import BatchSampler


class OffPolicyVectorizedSampler(BatchSampler):
    def __init__(self, algo, n_envs=None):
        super(OffPolicyVectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs

    @overrides
    def start_worker(self):
        n_envs = self.n_envs
        if n_envs is None:
            n_envs = int(self.algo.rollout_batch_size)
            n_envs = max(1, min(n_envs, 100))

        if getattr(self.algo.env, 'vectorized', False):
            self.vec_env = self.algo.env.vec_env_executor(
                n_envs=n_envs, max_path_length=self.algo.max_path_length)
        else:
            envs = [
                pickle.loads(pickle.dumps(self.algo.env))
                for _ in range(n_envs)
            ]
            self.vec_env = VecEnvExecutor(
                envs=envs, max_path_length=self.algo.max_path_length)
        self.env_spec = self.algo.env.spec

    @overrides
    def shutdown_worker(self):
        self.vec_env.close()

    @overrides
    def obtain_samples(self, itr):
        paths = []
        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs

        policy = self.algo.policy
        if self.algo.es:
            self.algo.es.reset()

        for rollout in range(self.algo.max_path_length):
            policy.reset(dones)
            if self.algo.replay_buffer_type == Buffer.HER:
                obs = [obs["observation"] for obs in obses]
                d_g = [obs["desired_goal"] for obs in obses]
                a_g = [obs["achieved_goal"] for obs in obses]
                input_obses = np.concatenate((obs, d_g), axis=-1)
            else:
                input_obses = obses
            if self.algo.es:
                actions, agent_infos = self.algo.es.get_actions(
                    input_obses, self.algo.policy)
            else:
                actions, agent_infos = self.algo.policy.get_actions(
                    input_obses)

            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            env_infos = tensor_utils.split_tensor_dict_list(env_infos)
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]

            if self.algo.replay_buffer_type == Buffer.REGULAR:
                self.algo.replay_buffer.add_transition(
                    observation=obses,
                    action=actions,
                    reward=rewards * self.algo.reward_scale,
                    terminal=dones,
                    next_observation=next_obses,
                )
            elif self.algo.replay_buffer_type == Buffer.HER:
                info_dict = {
                    "info_{}".format(key): []
                    for key in env_infos[0].keys()
                }
                for env_info in env_infos:
                    for key in env_info.keys():
                        info_dict["info_{}".format(key)].append(
                            env_info[key].reshape(1))
                self.algo.replay_buffer.add_transition(
                    observation=obs,
                    action=actions,
                    goal=d_g,
                    achieved_goal=a_g,
                    terminal=dones,
                    **info_dict,
                )

            if self.algo.replay_buffer_type == Buffer.HER and (
                    rollout == self.algo.max_path_length - 1):
                self.algo.replay_buffer.add_transition(
                    observation=[
                        next_obs['observation'] for next_obs in next_obses
                    ],
                    achieved_goal=[
                        next_obs['achieved_goal'] for next_obs in next_obses
                    ])

            for idx, reward, env_info, done in zip(itertools.count(), rewards,
                                                   env_infos, dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        rewards=[],
                        env_infos=[],
                    )
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)

                if self.algo.replay_buffer_type == Buffer.REGULAR:
                    add_to_paths = done or (
                        rollout == self.algo.max_path_length - 1)
                else:
                    add_to_paths = rollout == self.algo.max_path_length - 1
                if add_to_paths:
                    paths.append(
                        dict(
                            rewards=tensor_utils.stack_tensor_list(
                                running_paths[idx]["rewards"]),
                            env_infos=tensor_utils.stack_tensor_dict_list(
                                running_paths[idx]["env_infos"])))
                    running_paths[idx] = None

                    if self.algo.es:
                        self.algo.es.reset()
            obses = next_obses

        return paths

    @overrides
    def process_samples(self, itr, paths):
        success_history = []
        for path in paths:
            if "is_success" in path["env_infos"]:
                success = np.array(path["env_infos"]["is_success"])
                success_rate = np.mean(success)
                success_history.append(success_rate)

        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        samples_data = dict(
            undiscounted_returns=undiscounted_returns,
            success_history=success_history)
        return samples_data
