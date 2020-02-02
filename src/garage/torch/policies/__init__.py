"""PyTorch Policies."""
from garage.torch.policies.base import Policy
from garage.torch.policies.categorical_mlp_policy import Categorical_MLPPolicy
from garage.torch.policies.deterministic_mlp_policy import (
    DeterministicMLPPolicy)
from garage.torch.policies.gaussian_mlp_policy import GaussianMLPPolicy

__all__ = ['DeterministicMLPPolicy', 'GaussianMLPPolicy', 'Policy']
