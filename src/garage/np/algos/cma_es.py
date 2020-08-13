"""Covariance Matrix Adaptation Evolution Strategy."""
import collections

import cma
from dowel import logger, tabular
import numpy as np

from garage import EpisodeBatch, log_performance
from garage.np import paths_to_tensors
from garage.np.algos.rl_algorithm import RLAlgorithm
from garage.sampler import RaySampler


class CMAES(RLAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy.

    Note:
        The CMA-ES method can hardly learn a successful policy even for
        simple task. It is still maintained here only for consistency with
        original rllab paper.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.np.policies.Policy): Action policy.
        baseline (garage.np.baselines.Baseline): Baseline for GAE (Generalized
            Advantage Estimation).
        n_samples (int): Number of policies sampled in one epoch.
        discount (float): Environment reward discount.
        max_episode_length (int): Maximum length of a single episode.
        sigma0 (float): Initial std for param distribution.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 n_samples,
                 discount=0.99,
                 max_episode_length=500,
                 sigma0=1.):
        self.policy = policy
        self.max_episode_length = max_episode_length
        self.sampler_cls = RaySampler

        self._env_spec = env_spec
        self._discount = discount
        self._sigma0 = sigma0
        self._n_samples = n_samples
        self._baseline = baseline
        self._episode_reward_mean = collections.deque(maxlen=100)

        self._es = None
        self._all_params = None
        self._cur_params = None
        self._all_returns = None

    def _sample_params(self):
        """Return sample parameters.

        Returns:
            np.ndarray: A numpy array of parameter values.

        """
        return self._es.ask()

    def train(self, runner):
        """Initialize variables and start training.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        init_mean = self.policy.get_param_values()
        self._es = cma.CMAEvolutionStrategy(init_mean, self._sigma0,
                                            {'popsize': self._n_samples})
        self._all_params = self._sample_params()
        self._cur_params = self._all_params[0]
        self.policy.set_param_values(self._cur_params)
        self._all_returns = []

        # start actual training
        last_return = None

        for _ in runner.step_epochs():
            for _ in range(self._n_samples):
                runner.step_path = runner.obtain_samples(runner.step_itr)
                last_return = self.train_once(runner.step_itr,
                                              runner.step_path)
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
        # -- Stage: Calculate baseline
        if hasattr(self._baseline, 'predict_n'):
            baseline_predictions = self._baseline.predict_n(paths)
        else:
            baseline_predictions = [
                self._baseline.predict(path) for path in paths
            ]

        # -- Stage: Pre-process samples based on collected paths
        samples_data = paths_to_tensors(paths, self.max_episode_length,
                                        baseline_predictions, self._discount)

        # -- Stage: Run and calculate performance of the algorithm
        undiscounted_returns = log_performance(itr,
                                               EpisodeBatch.from_list(
                                                   self._env_spec, paths),
                                               discount=self._discount)
        self._episode_reward_mean.extend(undiscounted_returns)
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._episode_reward_mean))
        samples_data['average_return'] = np.mean(undiscounted_returns)

        epoch = itr // self._n_samples
        i_sample = itr - epoch * self._n_samples

        tabular.record('Epoch', epoch)
        tabular.record('# Sample', i_sample)

        rtn = samples_data['average_return']
        self._all_returns.append(samples_data['average_return'])

        if (itr + 1) % self._n_samples == 0:
            avg_rtns = np.array(self._all_returns)
            self._es.tell(self._all_params, -avg_rtns)
            self.policy.set_param_values(self._es.best.get()[0])

            # Clear for next epoch
            rtn = max(self._all_returns)
            self._all_returns.clear()
            self._all_params = self._sample_params()

        self._cur_params = self._all_params[(i_sample + 1) % self._n_samples]
        self.policy.set_param_values(self._cur_params)

        logger.log(tabular)
        return rtn
