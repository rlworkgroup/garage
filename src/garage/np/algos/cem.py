"""Cross Entropy Method."""
from dowel import logger, tabular
import numpy as np

from garage.np.algos import BatchPolopt


class CEM(BatchPolopt):
    """Cross Entropy Method.

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

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.np.policies.Policy): Action policy.
        baseline(garage.np.baselines.Baseline): Baseline for GAE
            (Generalized Advantage Estimation).
        n_samples (int): Number of policies sampled in one epoch.
        discount (float): Environment reward discount.
        max_path_length (int): Maximum length of a single rollout.
        best_frac (float): The best fraction.
        init_std (float): Initial std for policy param distribution.
        extra_std (float): Decaying std added to param distribution.
        extra_decay_time (float): Epochs that it takes to decay extra std.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 n_samples,
                 discount=0.99,
                 max_path_length=500,
                 init_std=1,
                 best_frac=0.05,
                 extra_std=1.,
                 extra_decay_time=100):
        super().__init__(policy, baseline, discount, max_path_length,
                         n_samples)
        self.env_spec = env_spec

        self.n_samples = n_samples
        self.best_frac = best_frac
        self.init_std = init_std
        self.best_frac = best_frac
        self.extra_std = extra_std
        self.extra_decay_time = extra_decay_time

    def _sample_params(self, epoch):
        extra_var_mult = max(1.0 - epoch / self.extra_decay_time, 0)
        sample_std = np.sqrt(
            np.square(self.cur_std) +
            np.square(self.extra_std) * extra_var_mult)
        return np.random.standard_normal(
            self.n_params) * sample_std + self.cur_mean

    def train(self, runner):
        """Initialize variables and start training.

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        Returns:
            The average return in last epoch cycle.

        """
        # epoch-wise
        self.cur_std = self.init_std
        self.cur_mean = self.policy.get_param_values()
        # epoch-cycle-wise
        self.cur_params = self.cur_mean
        self.all_returns = []
        self.all_params = [self.cur_mean.copy()]
        # constant
        self.n_best = int(self.n_samples * self.best_frac)
        assert self.n_best >= 1, (
            'n_samples is too low. Make sure that n_samples * best_frac >= 1')
        self.n_params = len(self.cur_mean)

        return super().train(runner)

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        """
        paths = self.process_samples(itr, paths)

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
        self.cur_params = self._sample_params(itr)
        self.all_params.append(self.cur_params.copy())
        self.policy.set_param_values(self.cur_params)

        logger.log(tabular)
        return rtn
