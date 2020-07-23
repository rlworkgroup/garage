#!/usr/bin/env python3
"""This is an example to add a simple VPG algorithm."""
import numpy as np
import torch

from garage import log_performance, TrajectoryBatch, wrap_experiment
from garage.envs import GarageEnv, PointEnv
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.misc import tensor_utils
from garage.sampler import RaySampler
from garage.torch.policies import GaussianMLPPolicy


# pylint: disable=too-few-public-methods
# pylint: disable=missing-return-doc, missing-return-type-doc
# pylint: disable=missing-class-docstring, missing-function-docstring
class SimpleVPG:  # noqa: D101

    sampler_cls = RaySampler

    def __init__(self, env_spec, policy):
        self.env_spec = env_spec
        self.policy = policy
        self.max_path_length = 200
        self._discount = 0.99
        self._policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def train(self, runner):  # noqa: D102
        for epoch in runner.step_epochs():
            samples = runner.obtain_samples(epoch)
            log_performance(
                epoch,
                TrajectoryBatch.from_trajectory_list(self.env_spec, samples),
                self._discount)
            self._train_once(samples)

    def _train_once(self, samples):
        losses = []
        self._policy_opt.zero_grad()
        for path in samples:
            returns_numpy = tensor_utils.discount_cumsum(
                path['rewards'], self._discount)
            returns = torch.Tensor(returns_numpy.copy())
            obs = torch.Tensor(path['observations'])
            actions = torch.Tensor(path['actions'])
            dist = self.policy(obs)[0]
            log_likelihoods = dist.log_prob(actions)
            loss = (-log_likelihoods * returns).mean()
            loss.backward()
            losses.append(loss.item())
        self._policy_opt.step()
        return np.mean(losses)


@wrap_experiment()
def debug_my_algorithm(ctxt=None):  # noqa: D103
    set_seed(100)
    runner = LocalRunner(ctxt)
    env = GarageEnv(PointEnv())
    # env = GarageEnv(env_name='LunarLanderContinuous-v2')
    policy = GaussianMLPPolicy(env.spec)
    algo = SimpleVPG(env.spec, policy)
    runner.setup(algo, env)
    runner.train(n_epochs=500, batch_size=4000, plot=True)


debug_my_algorithm()
