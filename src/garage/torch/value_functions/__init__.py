"""Value functions which use PyTorch."""
from garage.torch.value_functions.value_function import ValueFunction
from garage.torch.value_functions.gaussian_mlp_value_function import (  # noqa: I100,E501
    GaussianMLPValueFunction)

__all__ = ['ValueFunction', 'GaussianMLPValueFunction']
