"""PyTorch Policies."""
from garage.torch.policies.base import Policy
from garage.torch.policies.continuous_mlp_policy import ContinuousMLPPolicy
from garage.torch.policies.gaussian_mlp_policy import GaussianMLPPolicy

__all__ = ['ContinuousMLPPolicy', 'GaussianMLPPolicy', 'Policy']
