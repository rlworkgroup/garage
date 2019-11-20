"""PyTorch optimizers."""
from garage.torch.optimizers.conjugate_gradient_optimizer import (
    ConjugateGradientOptimizer)
from garage.torch.optimizers.differentiable_sgd import DiffSGD

__all__ = ['ConjugateGradientOptimizer', 'DiffSGD']
