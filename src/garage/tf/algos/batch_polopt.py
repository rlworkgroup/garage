"""Base class for batch sampling-based policy optimization methods."""

import collections

from dowel import logger, tabular
import numpy as np

from garage.misc import tensor_utils as np_tensor_utils
from garage.np.algos import RLAlgorithm
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.misc import tensor_utils
from garage.tf.samplers import BatchSampler


class BatchPolopt(RLAlgorithm):
    """Base class for batch sampling-based policy optimization methods.

    This includes various policy gradient methods like VPG, NPG, PPO, TRPO,
    etc.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.tf.policies.base.Policy): Policy.
        baseline (garage.tf.baselines.Baseline): The baseline.
        scope (str): Scope for identifying the algorithm.
            Must be specified if running multiple algorithms
            simultaneously, each using different environments
            and policies.
        max_path_length (int): Maximum length of a single rollout.
        discount (float): Discount.
        gae_lambda (float): Lambda used for generalized advantage
            estimation.
        center_adv (bool): Whether to rescale the advantages
            so that they have mean 0 and standard deviation 1.
        positive_adv (bool): Whether to shift the advantages
            so that they are always positive. When used in
            conjunction with center_adv the advantages will be
            standardized before shifting.
        fixed_horizon (bool): Whether to fix horizon.
        flatten_input (bool): Whether to flatten input along the observation
            dimension. If True, for example, an observation with shape (2, 4)
            will be flattened to 8.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 scope=None,
                 max_path_length=500,
                 discount=0.99,
                 gae_lambda=1,
                 center_adv=True,
                 positive_adv=False,
                 fixed_horizon=False,
                 flatten_input=True):
        self.env_spec = env_spec
        self.policy = policy
        self.baseline = baseline
        self.scope = scope
        self.max_path_length = max_path_length
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.fixed_horizon = fixed_horizon
        self.flatten_input = flatten_input

        self.episode_reward_mean = collections.deque(maxlen=100)
        if policy.vectorized:
            self.sampler_cls = OnPolicyVectorizedSampler
        else:
            self.sampler_cls = BatchSampler

        self.init_opt()

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            The average return in last epoch cycle.

        """
        last_return = None

        for epoch in runner.step_epochs():
            runner.step_path = runner.obtain_samples(runner.step_itr)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
        paths = self.process_samples(itr, paths)
        self.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, paths)
        return paths['average_return']

    def log_diagnostics(self, paths):
        """Log diagnostic information."""
        logger.log('Logging diagnostics...')
        self.policy.log_diagnostics(paths)
        self.baseline.log_diagnostics(paths)

    def process_samples(self, itr, paths):
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            dict: Processed sample data, with key
                * observations: (numpy.ndarray)
                * actions: (numpy.ndarray)
                * rewards: (numpy.ndarray)
                * baselines: (numpy.ndarray)
                * returns: (numpy.ndarray)
                * valids: (numpy.ndarray)
                * agent_infos: (dict)
                * env_infos: (dict)
                * paths: (list[dict])
                * average_return: (numpy.float64)

        """
        baselines = []
        returns = []

        max_path_length = self.max_path_length

        if self.flatten_input:
            paths = [
                dict(
                    observations=(self.env_spec.observation_space.flatten_n(
                        path['observations'])),
                    actions=(
                        self.env_spec.action_space.flatten_n(  # noqa: E126
                            path['actions'])),
                    rewards=path['rewards'],
                    env_infos=path['env_infos'],
                    agent_infos=path['agent_infos']) for path in paths
            ]
        else:
            paths = [
                dict(
                    observations=path['observations'],
                    actions=(
                        self.env_spec.action_space.flatten_n(  # noqa: E126
                            path['actions'])),
                    rewards=path['rewards'],
                    env_infos=path['env_infos'],
                    agent_infos=path['agent_infos']) for path in paths
            ]

        if hasattr(self.baseline, 'predict_n'):
            all_path_baselines = self.baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self.baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = (path['rewards'] + self.discount * path_baselines[1:] -
                      path_baselines[:-1])
            path['advantages'] = np_tensor_utils.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            path['deltas'] = deltas

        for idx, path in enumerate(paths):
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])

            # returns
            path['returns'] = np_tensor_utils.discount_cumsum(
                path['rewards'], self.discount)
            returns.append(path['returns'])

        # make all paths the same length
        obs = [path['observations'] for path in paths]
        obs = tensor_utils.pad_tensor_n(obs, max_path_length)

        actions = [path['actions'] for path in paths]
        actions = tensor_utils.pad_tensor_n(actions, max_path_length)

        rewards = [path['rewards'] for path in paths]
        rewards = tensor_utils.pad_tensor_n(rewards, max_path_length)

        returns = [path['returns'] for path in paths]
        returns = tensor_utils.pad_tensor_n(returns, max_path_length)

        baselines = tensor_utils.pad_tensor_n(baselines, max_path_length)

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

        average_discounted_return = (np.mean(
            [path['returns'][0] for path in paths]))

        undiscounted_returns = [sum(path['rewards']) for path in paths]
        self.episode_reward_mean.extend(undiscounted_returns)

        ent = np.sum(self.policy.distribution.entropy(agent_infos) *
                     valids) / np.sum(valids)

        samples_data = dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            valids=valids,
            agent_infos=agent_infos,
            env_infos=env_infos,
            paths=paths,
            average_return=np.mean(undiscounted_returns),
        )

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

    def init_opt(self):
        """Initialize the optimization procedure.

        If using tensorflow, this may include declaring all the variables and
        compiling functions.
        """
        raise NotImplementedError

    def get_itr_snapshot(self, itr):
        """Get all the data that should be saved in this snapshot iteration."""
        raise NotImplementedError

    def optimize_policy(self, itr, samples_data):
        """Optimize the policy using the samples."""
        raise NotImplementedError
