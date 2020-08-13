#!/usr/bin/env python3
"""This is an example to add a simple VPG algorithm."""
import numpy as np
import torch

from garage import EpisodeBatch, log_performance, wrap_experiment
from garage.envs import PointEnv
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.misc import tensor_utils
from garage.sampler import LocalSampler
from garage.torch.policies import GaussianMLPPolicy


# pylint: disable=too-few-public-methods
class SimpleVPG:
    """Simple Vanilla Policy Gradient.

    Args:
        env_spec (EnvSpec): Environment specification.
        policy (garage.tf.policies.StochasticPolicy): Policy.

    """
    sampler_cls = LocalSampler

    def __init__(self, env_spec, policy):
        self.env_spec = env_spec
        self.policy = policy
        self.max_episode_length = 200
        self._discount = 0.99
        self._policy_opt = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def train(self, runner):
        """Obtain samplers and start actual training for each epoch.

        Args:
            runner (LocalRunner): Experiment runner.

        """
        for epoch in runner.step_epochs():
            samples = runner.obtain_samples(epoch)
            log_performance(epoch,
                            EpisodeBatch.from_list(self.env_spec, samples),
                            self._discount)
            self._train_once(samples)

    def _train_once(self, samples):
        """Perform one step of policy optimization given one batch of samples.

        Args:
            samples (list[dict]): A list of collected paths.

        Returns:
            numpy.float64: Average return.

        """
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


@wrap_experiment
def tutorial_vpg(ctxt=None):
    """Train VPG with PointEnv environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~LocalRunner` to create the :class:`~Snapshotter`.

    """
    set_seed(100)
    runner = LocalRunner(ctxt)
    env = PointEnv()
    policy = GaussianMLPPolicy(env.spec)
    algo = SimpleVPG(env.spec, policy)
    runner.setup(algo, env)
    runner.train(n_epochs=200, batch_size=4000)


tutorial_vpg()
