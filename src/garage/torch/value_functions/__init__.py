"""Value functions which use PyTorch."""
from garage.torch.value_functions.gaussian_mlp_value_function import \
    GaussianMLPValueFunction  # noqa: I100,E501
from garage.torch.value_functions.value_function import ValueFunction

__all__ = ['ValueFunction', 'GaussianMLPValueFunction']
