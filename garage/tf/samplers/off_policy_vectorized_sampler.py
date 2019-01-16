"""
This module implements a Vectorized Sampler used for OffPolicy Algorithms.

It diffs from OnPolicyVectorizedSampler in two parts:
 - The num of envs is defined by rollout_batch_size. In
 OnPolicyVectorizedSampler, the number of envs can be decided by batch_size
 and max_path_length. But OffPolicy algorithms usually samples transitions
 from replay buffer, which only has buffer_batch_size.
 - It needs to add transitions to replay buffer throughout the rollout.
"""
import itertools
import pickle

import numpy as np

from garage.misc import tensor_utils
from garage.misc.overrides import overrides
from garage.tf.envs import VecEnvExecutor
from garage.tf.samplers import BatchSampler


class OffPolicyVectorizedSampler(BatchSampler):
    """This class implements OffPolicyVectorizedSampler."""

    def __init__(self, algo, n_envs=None):
        """
        Construct an OffPolicyVectorizedSampler.

        :param algo: Algorithms.
        :param n_envs: Number of parallelized sampling envs.
        """
        super(OffPolicyVectorizedSampler, self).__init__(algo)
        self.n_envs = n_envs

    @overrides
    def start_worker(self):
        """Initialize the sampler."""
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
        """Terminate workers if necessary."""
        self.vec_env.close()

    @overrides
    def obtain_samples(self, itr):
        """
        Collect samples for the given iteration number.

        :param itr: Iteration number.
        :return: A list of paths.
        """
        paths = []
        obses = self.vec_env.reset()
        dones = np.asarray([True] * self.vec_env.num_envs)
        running_paths = [None] * self.vec_env.num_envs
        n_samples = 0
        batch_samples = self.vec_env.num_envs * self.algo.max_path_length

        policy = self.algo.policy
        if self.algo.es:
            self.algo.es.reset()

        while n_samples < batch_samples:
            policy.reset(dones)
            if self.algo.input_include_goal:
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

            if self.algo.input_include_goal:
                self.algo.replay_buffer.add_transition(
                    observation=obs,
                    action=actions,
                    goal=d_g,
                    achieved_goal=a_g,
                    terminal=dones,
                    next_observation=[
                        next_obs["observation"] for next_obs in next_obses
                    ],
                    next_achieved_goal=[
                        next_obs["achieved_goal"] for next_obs in next_obses
                    ],
                )
            else:
                self.algo.replay_buffer.add_transition(
                    observation=obses,
                    action=actions,
                    reward=rewards * self.algo.reward_scale,
                    terminal=dones,
                    next_observation=next_obses,
                )

            for idx, reward, env_info, done in zip(itertools.count(), rewards,
                                                   env_infos, dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        rewards=[],
                        env_infos=[],
                    )
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)

                if done:
                    paths.append(
                        dict(
                            rewards=tensor_utils.stack_tensor_list(
                                running_paths[idx]["rewards"]),
                            env_infos=tensor_utils.stack_tensor_dict_list(
                                running_paths[idx]["env_infos"])))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None

                    if self.algo.es:
                        self.algo.es.reset()
            obses = next_obses

        return paths

    @overrides
    def process_samples(self, itr, paths):
        """
        Return processed sample data based on the collected paths.

        :param itr: Iteration number.
        :param paths: A list of collected paths.
        :return: Processed sample data.
        """
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
