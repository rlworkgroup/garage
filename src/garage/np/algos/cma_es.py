"""Covariance Matrix Adaptation Evolution Strategy."""
import cma
from dowel import logger, tabular
import numpy as np

from garage.np.algos import BatchPolopt


class CMAES(BatchPolopt):
    """Covariance Matrix Adaptation Evolution Strategy.

    Note:
        The CMA-ES method can hardly learn a successful policy even for
        simple task. It is still maintained here only for consistency with
        original rllab paper.

    Args:
        env_spec (garage.envs.EnvSpec): Environment specification.
        policy (garage.np.policies.Policy): Action policy.
        baseline (garage.np.baselines.Baseline): Baseline for GAE
            (Generalized Advantage Estimation).
        n_samples (int): Number of policies sampled in one epoch.
        discount (float): Environment reward discount.
        max_path_length (int): Maximum length of a single rollout.
        sigma0 (float): Initial std for param distribution.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 n_samples,
                 discount=0.99,
                 max_path_length=500,
                 sigma0=1.):
        super().__init__(env_spec=env_spec,
                         policy=policy,
                         baseline=baseline,
                         discount=discount,
                         max_path_length=max_path_length,
                         n_samples=n_samples)

        self.sigma0 = sigma0

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
        self._es = cma.CMAEvolutionStrategy(init_mean, self.sigma0,
                                            {'popsize': self.n_samples})
        self._all_params = self._sample_params()
        self._cur_params = self._all_params[0]
        self.policy.set_param_values(self._cur_params)
        self._all_returns = []

        return super().train(runner)

    def train_once(self, itr, paths):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            itr (int): Iteration number.
            paths (list[dict]): A list of collected paths.

        Returns:
            float: The average return in last epoch cycle.

        """
        paths = self.process_samples(itr, paths)

        epoch = itr // self.n_samples
        i_sample = itr - epoch * self.n_samples

        tabular.record('Epoch', epoch)
        tabular.record('# Sample', i_sample)

        rtn = paths['average_return']
        self._all_returns.append(paths['average_return'])

        if (itr + 1) % self.n_samples == 0:
            avg_rtns = np.array(self._all_returns)
            self._es.tell(self._all_params, -avg_rtns)
            self.policy.set_param_values(self._es.best.get()[0])

            # Clear for next epoch
            rtn = max(self._all_returns)
            self._all_returns.clear()
            self._all_params = self._sample_params()

        self._cur_params = self._all_params[(i_sample + 1) % self.n_samples]
        self.policy.set_param_values(self._cur_params)

        logger.log(tabular)
        return rtn
