"""A batch-based algorithm interleaves sampling and policy optimization."""
import abc
import collections

from dowel import tabular
import numpy as np

from garage import log_performance, TrajectoryBatch
from garage.misc import tensor_utils
from garage.np.algos.base import RLAlgorithm
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.samplers import BatchSampler


class BatchPolopt(RLAlgorithm):
    """A batch-based algorithm interleaves sampling and policy optimization.

    In one round of training, the runner will first instruct the sampler to do
    environment rollout and the sampler will collect a given number of samples
    (in terms of environment interactions). The collected paths are then
    absorbed by `RLAlgorithm.train_once()` and an algorithm performs one step
    of policy optimization. The updated policy will then be used in the
    next round of sampling.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        discount (float): Discount.
        max_path_length (int): Maximum length of a single rollout.
        n_samples (int): Number of train_once calls per epoch.

    """

    def __init__(self, env_spec, policy, baseline, discount, max_path_length,
                 n_samples):
        self.env_spec = env_spec
        self.policy = policy
        self.baseline = baseline
        self.discount = discount
        self.max_path_length = max_path_length
        self.n_samples = n_samples

        self.episode_reward_mean = collections.deque(maxlen=100)
        if policy.vectorized:
            self.sampler_cls = OnPolicyVectorizedSampler
        else:
            self.sampler_cls = BatchSampler

    @abc.abstractmethod
    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        last_return = None

        for _ in runner.step_epochs():
            for _ in range(self.n_samples):
                runner.step_path = runner.obtain_samples(runner.step_itr)
                tabular.record('TotalEnvSteps', runner.total_env_steps)
                last_return = self.train_once(runner.step_itr,
                                              runner.step_path)
                runner.step_itr += 1

        return last_return

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
            path['returns'] = tensor_utils.discount_cumsum(
                path['rewards'], self.discount)
            returns.append(path['returns'])

        obs = [path['observations'] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path['actions'] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path['rewards'] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        agent_infos = [path['agent_infos'] for path in paths]
        agent_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length)
            for p in agent_infos
        ])

        env_infos = [path['env_infos'] for path in paths]
        env_infos = tensor_utils.stack_tensor_dict_list([
            tensor_utils.pad_tensor_dict(p, max_path_length) for p in env_infos
        ])

        valids = [np.ones_like(path['returns']) for path in paths]
        valids = tensor_utils.pad_tensor_n(valids, max_path_length)

        ent = np.sum(self.policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        undiscounted_returns = log_performance(
            itr,
            TrajectoryBatch.from_trajectory_list(self.env_spec, paths),
            discount=self.discount)

        self.episode_reward_mean.extend(undiscounted_returns)

        tabular.record('Entropy', ent)
        tabular.record('Perplexity', np.exp(ent))
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self.episode_reward_mean))

        samples_data = dict(average_return=np.mean(undiscounted_returns))

        return samples_data
