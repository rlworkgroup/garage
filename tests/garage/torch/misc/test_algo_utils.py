"""Tests for utilities in torch/algos/_utils.py."""
import torch

from garage.torch.algos import compute_advantages


class TestAlgoUtils:
    """Tests for utilities in torch/algos/_utils.py."""

    def test_compute_advantages(self):  # noqa: D202
        """Test if compute_advantages() computes correctly."""

        def get_advantage(discount, rewards, baselines):
            adv = torch.zeros(rewards.shape)
            for i in range(rewards.shape[0]):
                acc = 0
                for j in range(rewards.shape[1]):
                    acc = acc * discount + rewards[i][-j - 1]
                    adv[i][-j - 1] = acc - baselines[i][-j - 1]
            return adv

        discount = 0.99
        length = 5

        # Tensor of shape (1, length)
        rewards = torch.rand(1, length)
        baselines = torch.rand(1, length)
        expected_adv = get_advantage(discount, rewards, baselines)
        computed_adv = compute_advantages(discount, 1, length, baselines,
                                          rewards)
        assert torch.allclose(expected_adv, computed_adv)

        # Tensor of shape (5, length)
        rewards = torch.rand(5, length)
        baselines = torch.rand(5, length)
        expected_adv = get_advantage(discount, rewards, baselines)
        computed_adv = compute_advantages(discount, 1, length, baselines,
                                          rewards)
        assert torch.allclose(expected_adv, computed_adv)
