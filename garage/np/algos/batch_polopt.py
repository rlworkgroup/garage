import collections

from dowel import tabular
import numpy as np

from garage.misc import special, tensor_utils
from garage.np.algos.base import RLAlgorithm


class BatchPolopt(RLAlgorithm):
    """A batch-based algorithm interleaves sampling and policy optimization.
    In one round of training, the runner will first instruct the sampler to do
    environment rollout and the sampler will collect a given number of samples
    (in terms of environment interactions). The collected paths are then
    absorbed by `RLAlgorithm.train_once()` and an algorithm performs one step
    of policy optimization.
    The updated policy will then be used in the next round of sampling.

    Args:
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        discount (float): Discount.
        max_path_length (int): Maximum length of a single rollout.
    """

    def __init__(self, policy, baseline, discount, max_path_length):
        self.policy = policy
        self.baseline = baseline
        self.discount = discount
        self.max_path_length = max_path_length

        self.episode_reward_mean = collections.deque(maxlen=100)

    def process_samples(self, itr, paths):
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths

        Returns:
            dict: Processed sample data, with key
                * average_return: (float)
        """
        baselines = []
        returns = []

        max_path_length = self.max_path_length

        if hasattr(self.baseline, 'predict_n'):
            all_path_baselines = self.baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self.baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path['returns'] = special.discount_cumsum(path['rewards'],
                                                      self.discount)
            returns.append(path['returns'])

        agent_infos = [path['agent_infos'] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        average_discounted_return = (np.mean(
            [path['returns'][0] for path in paths]))

        undiscounted_returns = [sum(path['rewards']) for path in paths]
        self.episode_reward_mean.extend(undiscounted_returns)

        ent = np.sum(self.policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        samples_data = dict(average_return=np.mean(undiscounted_returns))

        tabular.record('Iteration', itr)
        tabular.record('AverageDiscountedReturn', average_discounted_return)
        tabular.record('AverageReturn', np.mean(undiscounted_returns))
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self.episode_reward_mean))
        tabular.record('NumTrajs', len(paths))
        tabular.record('Entropy', ent)
        tabular.record('Perplexity', np.exp(ent))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))

        return samples_data
