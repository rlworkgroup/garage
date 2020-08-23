"""Value functions which use PyTorch."""
from garage.torch.value_functions.gaussian_mlp_value_function import (
    GaussianMLPValueFunction)
from garage.torch.value_functions.value_function import ValueFunction
from garage.torch.value_functions.continuous_mlp_value_function import ContinuousMLPValueFunction

__all__ = ['ContinuousMLPValueFunction', 'ValueFunction', 'GaussianMLPValueFunction']
