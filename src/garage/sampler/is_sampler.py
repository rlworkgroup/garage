import copy
from math import exp
from math import log
import random

from numpy import var

from garage.sampler import BatchSampler
from garage.sampler.utils import truncate_paths


class ISSampler(BatchSampler):
    """
    Sampler which alternates between live sampling iterations using
    BatchSampler and importance sampling iterations.

    Args:
        algo (garage.np.algos.RLAlgorithm): An algorithm instance.
        env (garage.envs.GarageEnv): An environement instance.
        n_backtrack (str/int): Number of past policies to update from
        n_is_pretrain (int): Number of importance sampling iterations to
            perform in beginning of training
        init_is (bool): Set initial iteration (after pretrain) an
            importance sampling iteration.
        skip_is_itrs (bool): Do not do any importance sampling
            iterations (after pretrain).
        hist_variance_penalty (int): Penalize variance of historical policy.
        max_is_ratio (int): Maximum allowed importance sampling ratio.
        ess_threshold (int): Minimum effective sample size required.
        randomize_draw (bool): Whether to randomize important samples.

    """

    def __init__(self,
                 algo,
                 env,
                 n_backtrack='all',
                 n_is_pretrain=0,
                 init_is=0,
                 skip_is_itrs=False,
                 hist_variance_penalty=0.0,
                 max_is_ratio=0,
                 ess_threshold=0,
                 randomize_draw=False):
        self.n_backtrack = n_backtrack
        self.n_is_pretrain = n_is_pretrain
        self.skip_is_itrs = skip_is_itrs

        self.hist_variance_penalty = hist_variance_penalty
        self.max_is_ratio = max_is_ratio
        self.ess_threshold = ess_threshold
        self.randomize_draw = randomize_draw

        self._hist = []
        self._is_itr = init_is

        super().__init__(algo, env)

    @property
    def history(self):
        """
        History of policies that have interacted with the environment and the
        data from interaction episode(s)
        """
        return self._hist

    def add_history(self, policy_distribution, paths):
        """
        Store policy distribution and paths in history
        """
        self._hist.append((policy_distribution, paths))

    def get_history_list(self, n_past='all'):
        """
        Get list of (distribution, data) tuples from history
        """
        if n_past == 'all':
            return self._hist
        return self._hist[-min(n_past, len(self._hist)):]

    def obtain_samples(self, itr, batch_size=None, whole_paths=True):
        # Importance sampling for first self.n_is_pretrain iterations
        if itr < self.n_is_pretrain:
            paths = self._obtain_is_samples(itr, batch_size, whole_paths)
            return paths

        # Alternate between importance sampling and live sampling
        if self._is_itr and not self.skip_is_itrs:
            paths = self._obtain_is_samples(itr, batch_size, whole_paths)
        else:
            paths = super().obtain_samples(itr, batch_size, whole_paths)
            if not self.skip_is_itrs:
                self.add_history(self.algo.policy.distribution, paths)

        self._is_itr = (self._is_itr + 1) % 2
        return paths

    def _obtain_is_samples(self, itr, batch_size=None, whole_paths=True):
        if batch_size is None:
            batch_size = self.algo.max_path_length

        paths = []
        for hist_policy_distribution, hist_paths in self.get_history_list(
                self.n_backtrack):
            h_paths = self._sample_isweighted_paths(
                policy=self.algo.policy,
                hist_policy_distribution=hist_policy_distribution,
                max_samples=batch_size,
                paths=hist_paths,
                hist_variance_penalty=self.hist_variance_penalty,
                max_is_ratio=self.max_is_ratio,
                ess_threshold=self.ess_threshold,
            )
            paths.extend(h_paths)

        if len(paths) > batch_size:
            paths = random.sample(paths, batch_size)

        return paths if whole_paths else truncate_paths(paths, batch_size)

    def _sample_isweighted_paths(self,
                                 policy,
                                 hist_policy_distribution,
                                 max_samples,
                                 paths=None,
                                 hist_variance_penalty=0.0,
                                 max_is_ratio=10,
                                 ess_threshold=0):
        if not paths:
            return []

        n_samples = min(len(paths), max_samples)

        samples = None
        if self.randomize_draw:
            samples = random.sample(paths, n_samples)
        elif paths:
            if n_samples == len(paths):
                samples = paths
            else:
                start = random.randint(0, len(paths) - n_samples)
                samples = paths[start:start + n_samples]

        # make duplicate of samples so we don't permanently alter historical
        # data
        samples = copy.deepcopy(samples)
        if ess_threshold > 0:
            is_weights = []

        dist1 = policy.distribution
        dist2 = hist_policy_distribution
        for path in samples:
            _, agent_infos = policy.get_actions(path['observations'])
            hist_agent_infos = path['agent_infos']
            if hist_variance_penalty > 0:
                hist_agent_infos['log_std'] += log(1.0 + hist_variance_penalty)
            path['agent_infos'] = agent_infos

            # compute importance sampling weight
            loglike_p = dist1.log_likelihood(path['actions'], agent_infos)
            loglike_hp = dist2.log_likelihood(path['actions'],
                                              hist_agent_infos)
            is_ratio = exp(sum(loglike_p) - sum(loglike_hp))

            # thresholding knobs
            if max_is_ratio > 0:
                is_ratio = min(is_ratio, max_is_ratio)
            if ess_threshold > 0:
                is_weights.append(is_ratio)

            # apply importance sampling weight
            path['rewards'] *= is_ratio

        if ess_threshold:
            # Effective sample size estimate.
            # Kong, Augustine. "A note on importance sampling using
            # standardized weights." University of Chicago, Dept.
            # of Statistics, Tech. Rep 348 (1992).
            if len(is_weights) / (1 + var(is_weights)) < ess_threshold:
                return []

        return samples
