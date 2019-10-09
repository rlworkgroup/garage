"""PyTorch Policies."""
from garage.torch.policies.base import Policy
from garage.torch.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy)
from garage.torch.policies.gaussian_mlp_policy import GaussianMLPPolicy
from garage.torch.policies.gaussian_mlp_policy_dual_head import GaussianMLPPolicyDualHead

__all__ = ['DeterministicMLPPolicy', 'GaussianMLPPolicy', 'GaussianMLPPolicyDualHead', 'Policy']
