import cma
import numpy as np

from garage.logger import logger, tabular
from garage.np.algos.base import RLAlgorithm


class CMAES(RLAlgorithm):
    r"""Covariance Matrix Adaptation Evolution Strategy.

    Note:
        The CMA-ES method can hardly learn a successful policy even for
        simple task. It is still maintained here only for consistency with
        original rllab paper.

    Attributes:
        env_spec(EnvSpec): Environment specification.
        policy(Policy): Action policy.
        baseline(): Baseline for GAE (Generalized Advantage Estimation).
        n_samples(int): Number of policies sampled in one epoch.
        gae_lambda(float): Lambda used for generalized advantage estimation.
        max_path_length(int):  Maximum length of a single rollout.
        discount(float): Environment reward discount.
        sigma0(float): Initial std for param distribution.

    """

    def __init__(self,
                 env_spec,
                 policy,
                 baseline,
                 n_samples,
                 gae_lambda=1,
                 max_path_length=500,
                 discount=0.99,
                 sigma0=1.):
        self.env_spec = env_spec
        self.policy = policy
        self.baseline = baseline
        self.n_samples = n_samples
        self.sigma0 = sigma0
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.max_path_length = max_path_length

        init_mean = self.policy.get_param_values()
        self.es = cma.CMAEvolutionStrategy(init_mean, sigma0)

        self.all_params = self.sample_params()
        self.cur_params = self.all_params[0]
        self.policy.set_param_values(self.cur_params)
        self.all_returns = []

    def sample_params(self):
        return self.es.ask(self.n_samples)

    def train_once(self, itr, paths):
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
