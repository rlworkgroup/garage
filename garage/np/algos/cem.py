import numpy as np

from garage.core import Serializable
from garage.logger import logger, tabular
from garage.np.algos.base import RLAlgorithm
from garage.plotter import Plotter


class CEM(RLAlgorithm, Serializable):
    r"""Cross Entropy Method.

    CEM works by iteratively optimizing a gaussian distribution of policy.

    In each epoch, CEM does the following:
    1. Sample n_samples policies from a gaussian distribution of
       mean cur_mean and std cur_std.
    2. Do rollouts for each policy.
    3. Update cur_mean and cur_std by doing Maximum Likelihood Estimation
       over the n_best top policies in terms of return.

    Note:
        When training CEM with LocalRunner, make sure that n_epoch_cycles for
        runner equals to n_samples for CEM.

        This implementation leverage n_epoch_cycles to do rollouts for a single
        policy in an epoch cycle.

    Attributes:
        env_spec(EnvSpec): Environment specification.
        policy(Policy): Action policy.
        baseline(): Baseline for GAE (Generalized Advantage Estimation).
        n_samples(int): Number of policies sampled in one epoch.
        max_path_length(int):  Maximum length of a single rollout.
        discount(float): Environment reward discount.
        init_std(float): Initial std for policy param distribution.
        extra_std(float): Decaying std added to param distribution.
        extra_decay_time(float): Epochs that it takes to decay extra std.
    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 n_samples,
                 gae_lambda=1,
                 max_path_length=500,
                 discount=0.99,
                 init_std=1,
                 best_frac=0.05,
                 extra_std=1.,
                 extra_decay_time=100,
                 **kwargs):
        self.quick_init(locals())
        self.env_spec = env_spec
        self.policy = policy
        self.baseline = baseline
        self.n_samples = n_samples
        self.extra_decay_time = extra_decay_time
        self.extra_std = extra_std
        self.best_frac = best_frac
        self.init_std = init_std
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.max_path_length = max_path_length
        self.plotter = Plotter()

        # epoch-wise
        self.cur_std = self.init_std
        self.cur_mean = self.policy.get_param_values()
        # epoch-cycle-wise
        self.cur_params = self.cur_mean
        self.all_returns = []
        self.all_params = [self.cur_mean.copy()]
        # fixed
        self.n_best = int(n_samples * best_frac)
        assert self.n_best >= 1, (
            f'n_samples is too low. Make sure that n_samples * best_frac >= 1')
        self.n_params = len(self.cur_mean)

    def sample_params(self, epoch):
        extra_var_mult = max(1.0 - epoch / self.extra_decay_time, 0)
        sample_std = np.sqrt(
            np.square(self.cur_std) +
            np.square(self.extra_std) * extra_var_mult)
        return np.random.standard_normal(
            self.n_params) * sample_std + self.cur_mean

    def train_once(self, itr, paths):
        epoch = itr // self.n_samples
        i_sample = itr - epoch * self.n_samples
        tabular.record('Epoch', epoch)
        tabular.record('# Sample', i_sample)
        # -- Stage: Process path
        rtn = paths['average_return']
        self.all_returns.append(paths['average_return'])

        # -- Stage: Update policy distribution.
        if (itr + 1) % self.n_samples == 0:
            avg_rtns = np.array(self.all_returns)
            best_inds = np.argsort(-avg_rtns)[:self.n_best]
            best_params = np.array(self.all_params)[best_inds]

            # MLE of normal distribution
            self.cur_mean = best_params.mean(axis=0)
            self.cur_std = best_params.std(axis=0)
            self.policy.set_param_values(self.cur_mean)

            # Clear for next epoch
            rtn = max(self.all_returns)
            self.all_returns.clear()
            self.all_params.clear()

        # -- Stage: Generate a new policy for next path sampling
        self.cur_params = self.sample_params(itr)
        self.all_params.append(self.cur_params.copy())
        self.policy.set_param_values(self.cur_params)

        logger.log(tabular)
        return rtn

    def get_itr_snapshot(self, itr):
        return dict(itr=itr, policy=self.policy, baseline=self.baseline)
