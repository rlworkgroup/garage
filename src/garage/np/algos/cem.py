"""Cross Entropy Method."""
import collections

from dowel import logger, tabular
import numpy as np

from garage import log_performance
from garage.np.algos.rl_algorithm import RLAlgorithm


class CEM(RLAlgorithm):
    """Cross Entropy Method.

    CEM works by iteratively optimizing a gaussian distribution of policy.

    In each epoch, CEM does the following:
    1. Sample n_samples policies from a gaussian distribution of
       mean cur_mean and std cur_std.
    2. Collect episodes for each policy.
    3. Update cur_mean and cur_std by doing Maximum Likelihood Estimation
       over the n_best top policies in terms of return.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.np.policies.Policy): Action policy.
        sampler (garage.sampler.Sampler): Sampler.
        n_samples (int): Number of policies sampled in one epoch.
        discount (float): Environment reward discount.
        best_frac (float): The best fraction.
        init_std (float): Initial std for policy param distribution.
        extra_std (float): Decaying std added to param distribution.
        extra_decay_time (float): Epochs that it takes to decay extra std.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 sampler,
                 n_samples,
                 discount=0.99,
                 init_std=1,
                 best_frac=0.05,
                 extra_std=1.,
                 extra_decay_time=100):
        self.policy = policy
        self.max_episode_length = env_spec.max_episode_length

        self._sampler = sampler

        self._best_frac = best_frac
        self._init_std = init_std
        self._extra_std = extra_std
        self._extra_decay_time = extra_decay_time
        self._episode_reward_mean = collections.deque(maxlen=100)
        self._env_spec = env_spec
        self._discount = discount
        self._n_samples = n_samples

        self._cur_std = None
        self._cur_mean = None
        self._cur_params = None
        self._all_returns = None
        self._all_params = None
        self._n_best = None
        self._n_params = None

    def _sample_params(self, epoch):
        """Return sample parameters.

        Args:
            epoch (int): Epoch number.

        Returns:
            np.ndarray: A numpy array of parameter values.

        """
        extra_var_mult = max(1.0 - epoch / self._extra_decay_time, 0)
        sample_std = np.sqrt(
            np.square(self._cur_std) +
            np.square(self._extra_std) * extra_var_mult)
        return np.random.standard_normal(
            self._n_params) * sample_std + self._cur_mean

    def train(self, trainer):
        """Initialize variables and start training.

        Args:
            trainer (Trainer): Experiment trainer, which provides services
                such as snapshotting and sampler control.

        Returns:
            float: The average return in last epoch cycle.

        """
        # epoch-wise
        self._cur_std = self._init_std
        self._cur_mean = self.policy.get_param_values()
        # epoch-cycle-wise
        self._cur_params = self._cur_mean
        self._all_returns = []
        self._all_params = [self._cur_mean.copy()]
        # constant
        self._n_best = int(self._n_samples * self._best_frac)
        assert self._n_best >= 1, (
            'n_samples is too low. Make sure that n_samples * best_frac >= 1')
        self._n_params = len(self._cur_mean)

        # start actual training
        last_return = None

        for _ in trainer.step_epochs():
            for _ in range(self._n_samples):
                trainer.step_episode = trainer.obtain_episodes(
                    trainer.step_itr)
                last_return = self._train_once(trainer.step_itr,
                                               trainer.step_episode)
                trainer.step_itr += 1

        return last_return

    def _train_once(self, itr, episodes):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            episodes (garage.EpisodeBatch): Episodes collected using the
                current policy.

        Returns:
            float: The average return of epoch cycle.

        """
        # -- Stage: Run and calculate performance of the algorithm
        undiscounted_returns = log_performance(itr,
                                               episodes,
                                               discount=self._discount)
        self._episode_reward_mean.extend(undiscounted_returns)
        tabular.record('Extras/EpisodeRewardMean',
                       np.mean(self._episode_reward_mean))
        average_return = np.mean(undiscounted_returns)

        epoch = itr // self._n_samples
        i_sample = itr - epoch * self._n_samples
        tabular.record('Epoch', epoch)
        tabular.record('# Sample', i_sample)
        rtn = average_return
        self._all_returns.append(average_return)

        # -- Stage: Update policy distribution.
        if (itr + 1) % self._n_samples == 0:
            avg_rtns = np.array(self._all_returns)
            best_inds = np.argsort(-avg_rtns)[:self._n_best]
            best_params = np.array(self._all_params)[best_inds]

            # MLE of normal distribution
            self._cur_mean = best_params.mean(axis=0)
            self._cur_std = best_params.std(axis=0)
            self.policy.set_param_values(self._cur_mean)

            # Clear for next epoch
            rtn = max(self._all_returns)
            self._all_returns.clear()
            self._all_params.clear()

        # -- Stage: Generate a new policy for next path sampling
        self._cur_params = self._sample_params(itr)
        self._all_params.append(self._cur_params.copy())
        self.policy.set_param_values(self._cur_params)

        logger.log(tabular)
        return rtn
