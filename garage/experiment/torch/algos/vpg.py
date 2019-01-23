import numpy as np
import torch

from garage.experiment import Agent
from garage.experiment.core import Policy
from garage.experiment.loggers import Summary
from garage.misc.special import discount_cumsum, explained_variance_1d


class VPG(Agent):
    """
    Vanilla Policy Gradient.

    TODO: Implement baseline support.
          Testing.
    """

    def __init__(self, env_spec, policy: Policy, discount, baseline, *args,
                 **kwargs):

        self.env_spec = env_spec
        self.policy = policy
        self.discount = discount
        self.baseline = baseline

        self.policy_pi_opt = torch.optim.Adam(self.policy.parameters())

        self.stats = Summary()

    def get_actions(self, obs):
        self.policy.eval()
        actions, _ = self.policy.sample(obs)
        return actions

    def train_once(self, paths):
        self.stats.reset()

        logp_pi, adv = self._process_sample(paths)

        pi_loss = -(logp_pi * adv).mean()
        self.policy.train()
        self.policy_pi_opt.zero_grad()
        pi_loss.backward()
        self.policy_pi_opt.step()

    def _process_sample(self, paths):
        self.policy.eval()
        logp_pi_all = torch.empty((0, ))
        adv_all = np.array([], dtype=np.float32)

        # For stats
        rtns_all = np.array([])
        undiscounted_rtns_all = np.array([])
        predicted_rtns_all = np.array([])

        # Add 'return' to paths required by baseline
        for path in paths:
            rews = path['rewards']

            rtns = discount_cumsum(rews, self.discount)
            undiscounted_rtns = discount_cumsum(rews, 1)

            path['returns'] = rtns
            rtns_all = np.append(rtns_all, rtns)
            undiscounted_rtns_all = np.append(undiscounted_rtns_all,
                                              undiscounted_rtns)

        self.baseline.fit(paths)

        for path in paths:
            obs = torch.Tensor(path['observations'])
            actions = torch.Tensor(path['actions']).view(-1, 1)
            logp_pi = self.policy._logpdf(obs, actions)

            rtns = path['returns']
            baselines = self.baseline.predict(path)
            advs = rtns - baselines

            logp_pi_all = torch.cat((logp_pi_all, logp_pi))
            adv_all = np.concatenate((adv_all, advs))
            predicted_rtns_all = np.append(predicted_rtns_all, baselines)

        # Save stats
        self.stats.scalar('AverageDiscountedReturn', rtns_all.mean())
        self.stats.scalar('AverageReturn', undiscounted_rtns_all.mean())
        self.stats.scalar('ExplainedVariance',
                          explained_variance_1d(predicted_rtns_all, rtns_all))
        self.stats.scalar('StdReturn', rtns_all.std())
        self.stats.scalar('MinReturn', rtns_all.min())
        self.stats.scalar('MaxReturn', rtns_all.max())

        return logp_pi_all, torch.Tensor(adv_all)

    def get_summary(self):
        return self.stats.copy()
