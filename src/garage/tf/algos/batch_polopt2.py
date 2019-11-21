"""Base class for batch sampling-based policy optimization methods."""
import abc
import collections

from dowel import logger, tabular
import numpy as np

from garage.misc import tensor_utils as np_tensor_utils
from garage.np.algos import RLAlgorithm
from garage.sampler import OnPolicyVectorizedSampler
from garage.tf.samplers import BatchSampler


class BatchPolopt2(RLAlgorithm, abc.ABC):
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
        self._env_spec = env_spec
        self._policy = policy
        self._baseline = baseline
        self._scope = scope
        self._max_path_length = max_path_length
        self._discount = discount
        self._gae_lambda = gae_lambda
        self._center_adv = center_adv
        self._positive_adv = positive_adv
        self._fixed_horizon = fixed_horizon
        self._flatten_input = flatten_input

        self._episode_reward_mean = collections.deque(maxlen=100)

        if policy.vectorized:
            self._sampler_cls = OnPolicyVectorizedSampler
        else:
            self._sampler_cls = BatchSampler
        self.init_opt()

    @property
    def max_path_length(self):
        """Max path length of samples.

        Returns:
            int: Max path length.

        """
        return self._max_path_length

    @property
    def policy(self):
        """Policy used by the algorithm.

        Returns:
            garage.tf.policies: Policy.

        """
        return self._policy

    @property
    def sampler_cls(self):
        """Sampler class used by the algorith.

        Returns:
            garage.sampler: Sampler.

        """
        return self._sampler_cls

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
            runner.step_path = runner.obtain_samples(runner.step_itr)
            last_return = self.train_once(runner.step_itr, runner.step_path)
            runner.step_itr += 1

        return last_return

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            float: The average return in last epoch cycle.

        """
        paths = self.process_samples(itr, paths)
        self.log_diagnostics(paths)
        logger.log('Optimizing policy...')
        self.optimize_policy(itr, paths)
        return paths['average_return']

    def log_diagnostics(self, paths):
        """Log diagnostic information.

        Args:
            paths (list[dict]): A list of collected paths.

        """
        logger.log('Logging diagnostics...')
        self._policy.log_diagnostics(paths)
        self._baseline.log_diagnostics(paths)

    def process_samples(self, itr, paths):
        """Return processed sample data based on the collected paths.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            dict: Processed sample data.

        Note:
            The returned samples is a dictionary with keys
                - observations: (numpy.ndarray), shape [B * (T), *obs_dims]
                - actions: (numpy.ndarray), shape [B * (T), *act_dims]
                - rewards : (numpy.ndarray), shape [B * (T), ]
                - baselines: (numpy.ndarray), shape [B * (T), ]
                - returns: (numpy.ndarray), shape [B * (T), ]
                - lengths: (numpy.ndarray), shape [P, ], i-th entry represents
                  the length of i-th path.
                - valids: (numpy.ndarray), shape [P, ], [i, j] entry is 1 if
                  the j-th sample in i-th path is valid, otherwise 0.
                - agent_infos: (dict), see
                  OnPolicyVectorizedSampler.obtain_samples()
                - env_infos: (dict), see
                  OnPolicyVectorizedSampler.obtain_samples()
                - paths: (list[dict]) The original path with observation or
                  action flattened
                - average_return: (numpy.float64)

            where B = batch size, (T) = variable-length of each trajectory,
            P = number of paths. Notice that B * T equals to the total number
            of environment steps in all trajectories.

        """
        baselines = []
        returns = []

        if self._flatten_input:
            paths = [
                dict(
                    observations=(self._env_spec.observation_space.flatten_n(
                        path['observations'])),
                    actions=(
                        self._env_spec.action_space.flatten_n(  # noqa: E126
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
                        self._env_spec.action_space.flatten_n(  # noqa: E126
                            path['actions'])),
                    rewards=path['rewards'],
                    env_infos=path['env_infos'],
                    agent_infos=path['agent_infos']) for path in paths
            ]

        if hasattr(self._baseline, 'predict_n'):
            all_path_baselines = self._baseline.predict_n(paths)
        else:
            all_path_baselines = [
                self._baseline.predict(path) for path in paths
            ]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = (path['rewards'] + self._discount * path_baselines[1:] -
                      path_baselines[:-1])
            path['advantages'] = np_tensor_utils.discount_cumsum(
                deltas, self._discount * self._gae_lambda)
            path['deltas'] = deltas
            # baselines
            path['baselines'] = all_path_baselines[idx]
            baselines.append(path['baselines'])
            # returns
            path['returns'] = np_tensor_utils.discount_cumsum(
                path['rewards'], self._discount)
            returns.append(path['returns'])

        obs = np.concatenate([path['observations'] for path in paths])
        actions = np.concatenate([path['actions'] for path in paths])
        rewards = np.concatenate([path['rewards'] for path in paths])
        returns = np.concatenate(returns)
        baselines = np.concatenate(baselines)

        agent_infos_path = [path['agent_infos'] for path in paths]
        agent_infos = dict()
        for key in self._policy.state_info_keys:
            agent_infos[key] = np.concatenate(
                [infos[key] for infos in agent_infos_path])

        env_infos_path = [path['env_infos'] for path in paths]
        env_infos = dict()
        for key in paths[0]['env_infos'].keys():
            env_infos[key] = np.concatenate(
                [infos[key] for infos in env_infos_path])

        valids = np.asarray([np.ones_like(path['returns']) for path in paths])
        lengths = np.asarray([v.sum() for v in valids])

        average_discounted_return = (np.mean(
            [path['returns'][0] for path in paths]))

        undiscounted_returns = [sum(path['rewards']) for path in paths]
        self._episode_reward_mean.extend(undiscounted_returns)

        samples_data = dict(
            observations=obs,
            actions=actions,
            rewards=rewards,
            baselines=baselines,
            returns=returns,
            lengths=lengths,
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
                       np.mean(self._episode_reward_mean))
        tabular.record('NumTrajs', len(paths))
        tabular.record('StdReturn', np.std(undiscounted_returns))
        tabular.record('MaxReturn', np.max(undiscounted_returns))
        tabular.record('MinReturn', np.min(undiscounted_returns))

        return samples_data

    @abc.abstractmethod
    def init_opt(self):
        """Initialize the optimization procedure.

        If using tensorflow, this may include declaring all the variables and
        compiling functions.
        """

    @abc.abstractmethod
    def get_itr_snapshot(self, itr):
        """Get all the data that should be saved in this snapshot iteration.

        Args:
            itr (int): Iteration.

        """

    @abc.abstractmethod
    def optimize_policy(self, itr, samples_data):
        """Optimize the policy using the samples.

        Args:
            itr (int): Iteration number.
            samples_data (dict[numpy.ndarray]): Sample data.

        """
