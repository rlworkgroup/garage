#!/usr/bin/env python3
"""This is an example to add a Cross Entropy Method algorithm."""
import numpy as np

from garage import log_performance, TrajectoryBatch, wrap_experiment
from garage.envs import GarageEnv
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
from garage.misc import tensor_utils
from garage.sampler import RaySampler
from garage.tf.policies import CategoricalMLPPolicy


# pylint: disable=too-few-public-methods
# pylint: disable=missing-return-doc, missing-return-type-doc
# pylint: disable=missing-class-docstring, missing-function-docstring
class SimpleCEM:  # noqa: D101

    sampler_cls = RaySampler

    def __init__(self, env_spec, policy):
        self.env_spec = env_spec
        self.policy = policy
        self.max_path_length = 200
        self._discount = 0.99
        self._extra_std = 1
        self._extra_decay_time = 100
        self._n_samples = 20
        self._n_best = 1
        self._cur_std = 1
        self._cur_mean = self.policy.get_param_values()
        self._all_avg_returns = []
        self._all_params = [self._cur_mean.copy()]
        self._cur_params = None

    def train(self, runner):  # noqa: D102
        for epoch in runner.step_epochs():
            samples = runner.obtain_samples(epoch)
            log_performance(
                epoch,
                TrajectoryBatch.from_trajectory_list(self.env_spec, samples),
                self._discount)
            self._train_once(epoch, samples)

    def _train_once(self, epoch, paths):
        returns = []
        for path in paths:
            returns.append(
                tensor_utils.discount_cumsum(path['rewards'], self._discount))
        avg_return = np.mean(np.concatenate(returns))
        self._all_avg_returns.append(avg_return)
        if (epoch + 1) % self._n_samples == 0:
            avg_rtns = np.array(self._all_avg_returns)
            best_inds = np.argsort(-avg_rtns)[:self._n_best]
            best_params = np.array(self._all_params)[best_inds]
            self._cur_mean = best_params.mean(axis=0)
            self._cur_std = best_params.std(axis=0)
            self.policy.set_param_values(self._cur_mean)
            avg_return = max(self._all_avg_returns)
            self._all_avg_returns.clear()
            self._all_params.clear()
        self._cur_params = self._sample_params(epoch)
        self._all_params.append(self._cur_params.copy())
        self.policy.set_param_values(self._cur_params)
        return avg_return

    def _sample_params(self, epoch):
        extra_var_mult = max(1.0 - epoch / self._extra_decay_time, 0)
        sample_std = np.sqrt(
            np.square(self._cur_std) +
            np.square(self._extra_std) * extra_var_mult)
        return np.random.standard_normal(len(
            self._cur_mean)) * sample_std + self._cur_mean


@wrap_experiment()
def debug_my_algorithm(ctxt=None):  # noqa: D103
    set_seed(100)
    with LocalTFRunner(ctxt) as runner:
        env = GarageEnv(env_name='CartPole-v1')
        policy = CategoricalMLPPolicy(env.spec)
        algo = SimpleCEM(env.spec, policy)
        runner.setup(algo, env)
        runner.train(n_epochs=500, batch_size=4000, plot=True)


debug_my_algorithm()
