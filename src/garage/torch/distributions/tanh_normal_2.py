from collections import namedtuple

import torch
from torch.distributions import Distribution, MultivariateNormal


class TanhNormal2(Distribution):
    def __init__(self, mean, cov, epsilon=1e-6):
        self.normal_mean = mean
        self.normal_std = cov
        self.normal = MultivariateNormal(mean, cov)
        self.epsilon = epsilon

    def log_prob(self, value, pre_tanh_value=None):
        """
        pre_tanh_value should usually not be None, but as the option available just in case a value is not passed.
        """
        def clip_but_pass_gradient(x, l=-1., u=1.):
            clip_up = (x > u).float()
            clip_low = (x < l).float()
            with torch.no_grad():
                clip = ((u - x)*clip_up + (l - x)*clip_low)
            return x + clip

        if pre_tanh_value is None:
            pre_tanh_value = torch.log((1+value) / (1-value)) / 2
        ret = self.normal.log_prob(pre_tanh_value) - torch.sum(torch.log(clip_but_pass_gradient((1. - value**2), l=0., u=1.) + self.epsilon), axis=-1)
        return ret

    def sample(self, return_pre_tanh_value=False):
        """
        Gradients will and should *not* pass through this operation.
        """
        with torch.no_grad():
            z = self.normal.sample()
            if return_pre_tanh_value:
                action_infos = namedtuple("action_infos", ["pre_tanh_action",
                                                           "action"])
                return action_infos(z, torch.tanh(z))
            return torch.tanh(z)

    def rsample(self, return_pre_tanh_value=False):
        """
        Sampling in the reparameterization case.
        """
        z = self.normal.rsample()
        if return_pre_tanh_value:
            action_infos = namedtuple("action_infos", ["pre_tanh_action",
                                                       "action"])
            return action_infos(z, torch.tanh(z))
        return torch.tanh(z)

    @property
    def loc(self):
        return self.normal.loc

    @property
    def mean(self):
        return torch.tanh(self.normal.mean)

    @property
    def variance(self):
        return self.normal.variance

    def entropy(self):
        return self.normal.entropy()
