"""Covariance Matrix Adaptation Evolution Strategy."""
import collections

import cma
from dowel import logger, tabular
import numpy as np

from garage import log_performance
from garage.np.algos.rl_algorithm import RLAlgorithm


class CMAES(RLAlgorithm):
    """Covariance Matrix Adaptation Evolution Strategy.

    Note:
        The CMA-ES method can hardly learn a successful policy even for
        simple task. It is still maintained here only for consistency with
        original rllab paper.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.np.policies.Policy): Action policy.
        sampler (garage.sampler.Sampler): Sampler.
        n_samples (int): Number of policies sampled in one epoch.
        discount (float): Environment reward discount.
        sigma0 (float): Initial std for param distribution.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 sampler,
                 n_samples,
                 discount=0.99,
                 sigma0=1.):
        self.policy = policy
        self.max_episode_length = env_spec.max_episode_length
        self._sampler = sampler

        self._env_spec = env_spec
        self._discount = discount
        self._sigma0 = sigma0
        self._n_samples = n_samples
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

    def train(self, trainer):
        """Initialize variables and start training.

        Args:
            trainer (Trainer): Trainer is passed to give algorithm
                the access to trainer.step_epochs(), which provides services
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
