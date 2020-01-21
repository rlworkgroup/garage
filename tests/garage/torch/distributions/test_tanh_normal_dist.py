import pytest
import torch

from garage.torch.distributions import TanhNormal2


class TestBenchmarkTanhNormalDistribution:

    def test_new_tanh_normal(self):
        mean = torch.ones(1)
        std = torch.ones(1)
        cov = (std**2).diag_embed()
        dist = TanhNormal2(mean, cov)
        del dist

    def test_tanh_normal_bounds(self):
        mean = torch.ones(1) * 100
        std = torch.ones(1) * 100
        cov = (std**2).diag_embed()
        dist = TanhNormal2(mean, cov)
        assert dist.mean <= 1.
        del dist
        mean = torch.ones(1) * -100
        std = torch.ones(1) * 100
        cov = (std**2).diag_embed()
        dist = TanhNormal2(mean, cov)
        assert dist.mean >= -1.

    def test_tanh_normal_rsample(self):
        mean = torch.zeros(1)
        std = torch.ones(1)
        cov = (std**2).diag_embed()
        dist = TanhNormal2(mean, cov)
        sample = dist.rsample(return_pre_tanh_value=True)
        assert sample.pre_tanh_action is not None and sample.action is not None
        assert (sample.pre_tanh_action.tanh() == sample.action).all()
        sample_without_pre_tanh = dist.rsample()
        assert -1 <= sample_without_pre_tanh <= 1.
        assert -1 <= sample.action <= 1.

    def test_tanh_normal_sample(self):
        mean = torch.zeros(1)
        std = torch.ones(1)
        cov = (std**2).diag_embed()
        dist = TanhNormal2(mean, cov)
        sample = dist.sample(return_pre_tanh_value=True)
        assert sample.pre_tanh_action is not None and sample.action is not None
        assert (sample.pre_tanh_action.tanh() == sample.action).all()
        sample_without_pre_tanh = dist.sample()
        assert -1 <= sample_without_pre_tanh <= 1.
        assert -1 <= sample.action <= 1.

    def test_tanh_normal_log_prob(self):
        mean = torch.zeros(1)
        std = torch.ones(1)
        cov = (std**2).diag_embed()
        dist = TanhNormal2(mean, cov)
        pre_tanh_action = torch.Tensor([[2.0960]])
        action = pre_tanh_action.tanh()
        log_prob = dist.log_prob(action, pre_tanh_action)
        log_prob_approx = dist.log_prob(action)
        assert torch.allclose(log_prob, torch.Tensor([-0.2798519]))
        assert torch.allclose(log_prob_approx, torch.Tensor([-0.2798519]))
