from collections import deque

import numpy as np
import tensorflow as tf

from garage.misc import logger, special
from garage.sampler import parallel_sampler
from garage.sampler import singleton_pool
from garage.sampler.base import BaseSampler
from garage.sampler.utils import truncate_paths
from garage.tf.misc import tensor_utils


def worker_init_tf(g):
    g.sess = tf.Session()
    g.sess.__enter__()


def worker_init_tf_vars(g):
    g.sess.run(tf.global_variables_initializer())


class BatchSampler(BaseSampler):
    def __init__(self, algo, env, n_envs):
        super(BatchSampler, self).__init__(algo, env)
        self.n_envs = n_envs
        self.eprewmean = deque(maxlen=100)

    def start_worker(self):
        assert singleton_pool.initialized, \
            "Use singleton_pool.initialize(n_parallel) to setup workers."
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf)
        parallel_sampler.populate_task(self.env, self.algo.policy)
        if singleton_pool.n_parallel > 1:
            singleton_pool.run_each(worker_init_tf_vars)

    def shutdown_worker(self):
        parallel_sampler.terminate_task(scope=self.algo.scope)

    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        if not batch_size:
            batch_size = self.algo.max_path_length * self.n_envs

        cur_policy_params = self.algo.policy.get_param_values()
        paths = parallel_sampler.sample_paths(
            policy_params=cur_policy_params,
            max_samples=batch_size,
            max_path_length=self.algo.max_path_length,
            scope=self.algo.scope,
        )
        if whole_paths:
            return paths
        else:
            paths_truncated = truncate_paths(paths, batch_size)
            return paths_truncated

    def process_samples(self, itr, paths):
        baselines = []
        returns = []

        max_path_length = self.algo.max_path_length

        if hasattr(self.algo.baseline, "predict_n"):
            all_path_baselines = self.algo.baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self.algo.baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] \
                + self.algo.discount * path_baselines[1:] \
                - path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.algo.discount * self.algo.gae_lambda)
            path["deltas"] = deltas

        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path["returns"] = special.discount_cumsum(path["rewards"],
                                                      self.algo.discount)
            returns.append(path["returns"])

        # make all paths the same length
        obs = [path["observations"] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path["actions"] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path["rewards"] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path["returns"] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        advantages = [path["advantages"] for path in paths]
        advantages = tensor_utils.pad_tensor_n(advantages, max_path_length)

        baselines = tensor_utils.pad_tensor_n(baselines, max_path_length)

        agent_infos = [path["agent_infos"] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        env_infos = [path["env_infos"] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos
        ])

        valids = [np.ones_like(path["returns"]) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        average_discounted_return = (np.mean(
            [path["returns"][0] for path in paths]))

        undiscounted_returns = [sum(path["rewards"]) for path in paths]
        self.eprewmean.extend(undiscounted_returns)

        ent = np.sum(
            self.algo.policy.distribution.entropy(agent_infos) *
            valids) / np.sum(valids)

        samples_data = dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            advantages=advantages,
            baselines=baselines,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            env_infos=env_infos,
            paths=paths,
            average_return=np.mean(undiscounted_returns),
        )

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        logger.record_tabular('AverageReturn', np.mean(undiscounted_returns))
        logger.record_tabular('Extras/EpisodeRewardMean',
                              np.mean(self.eprewmean))
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular('StdReturn', np.std(undiscounted_returns))
        logger.record_tabular('MaxReturn', np.max(undiscounted_returns))
        logger.record_tabular('MinReturn', np.min(undiscounted_returns))

        return samples_data
