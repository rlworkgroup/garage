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
        super().__init__(policy, baseline, discount, max_path_length,
                         n_samples)
        self.env_spec = env_spec
        self.policy = policy

        self.sigma0 = sigma0

        init_mean = self.policy.get_param_values()
        self.es = cma.CMAEvolutionStrategy(init_mean, sigma0)

        self.all_params = self.sample_params()
        self.cur_params = self.all_params[0]
        self.policy.set_param_values(self.cur_params)
        self.all_returns = []

    def sample_params(self):
        return self.es.ask(self.n_samples)

    def train_once(self, itr, paths):
        paths = self.process_samples(itr, paths)

        epoch = itr // self.n_samples
        i_sample = itr - epoch * self.n_samples

        tabular.record('Epoch', epoch)
        tabular.record('# Sample', i_sample)

        rtn = paths['average_return']
        self.all_returns.append(paths['average_return'])

        if (itr + 1) % self.n_samples == 0:
            avg_rtns = np.array(self.all_returns)
            self.es.tell(self.all_params, -avg_rtns)
            self.policy.set_param_values(self.es.result()[0])

            # Clear for next epoch
            rtn = max(self.all_returns)
            self.all_returns.clear()
            self.all_params = self.sample_params()

        self.cur_params = self.all_params[(i_sample + 1) % self.n_samples]
        self.policy.set_param_values(self.cur_params)

        logger.log(tabular)
        return rtn

    def get_itr_snapshot(self, itr):
        return dict(itr=itr, policy=self.policy, baseline=self.baseline)
