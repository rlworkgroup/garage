"""Tests for the tanh transformed normal distribution."""
import torch

from garage.torch.distributions import TanhNormal


class TestBenchmarkTanhNormalDistribution:
    """Tests for the tanh normal distribution."""

    def test_new_tanh_normal(self):
        """Tests the tanh_normal constructor."""
        mean = torch.ones(1)
        std = torch.ones(1)
        dist = TanhNormal(mean, std)
        del dist

    def test_tanh_normal_bounds(self):
        """Test to make sure the tanh_normal dist obeys the bounds (-1,1)."""
        mean = torch.ones(1) * 100
        std = torch.ones(1) * 100
        dist = TanhNormal(mean, std)
        assert dist.mean <= 1.
        del dist
        mean = torch.ones(1) * -100
        std = torch.ones(1) * 100
        dist = TanhNormal(mean, std)
        assert dist.mean >= -1.

    def test_tanh_normal_rsample(self):
        """Test the bounds of the tanh_normal rsample function."""
        mean = torch.zeros(1)
        std = torch.ones(1)
        dist = TanhNormal(mean, std)
        sample = dist.rsample()
        pre_tanh_action, action = dist.rsample_with_pre_tanh_value()
        assert (pre_tanh_action.tanh() == action).all()
        assert -1 <= action <= 1.
        assert -1 <= sample <= 1.
        del dist

    def test_tanh_normal_log_prob(self):
        """Verify the correctnes of the tanh_normal log likelihood function."""
        mean = torch.zeros(1)
        std = torch.ones(1)
        dist = TanhNormal(mean, std)
        pre_tanh_action = torch.Tensor([[2.0960]])
        action = pre_tanh_action.tanh()
        log_prob = dist.log_prob(action, pre_tanh_action)
        log_prob_approx = dist.log_prob(action)
        assert torch.allclose(log_prob, torch.Tensor([-0.2798519]))
        assert torch.allclose(log_prob_approx, torch.Tensor([-0.2798519]))
        del dist

    def test_tanh_normal_expand(self):
        """Test for expand function.

        Checks whether expand returns a distribution that has potentially a
        different batch size from the already existing distribution.

        """
        mean = torch.zeros(1)
        std = torch.ones(1)
        dist = TanhNormal(mean, std)
        new_dist = dist.expand((2, ))
        sample = new_dist.sample()
        assert sample.shape == torch.Size((2, 1))

    def test_tanh_normal_repr(self):
        """Test that the repr function outputs the class name."""
        mean = torch.zeros(1)
        std = torch.ones(1)
        dist = TanhNormal(mean, std)
        assert repr(dist) == 'TanhNormal'
