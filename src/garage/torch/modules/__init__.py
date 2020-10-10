"""PyTorch Modules."""
# yapf: disable
# isort:skip_file
from garage.torch.modules.categorical_cnn_module import CategoricalCNNModule
from garage.torch.modules.cnn_module import CNNModule
from garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPIndependentStdModule)  # noqa: E501
from garage.torch.modules.gaussian_mlp_module import (
    GaussianMLPTwoHeadedModule)  # noqa: E501
from garage.torch.modules.gaussian_mlp_module import GaussianMLPModule
from garage.torch.modules.mlp_module import MLPModule
from garage.torch.modules.multi_headed_mlp_module import MultiHeadedMLPModule
# DiscreteCNNModule must go after MLPModule
from garage.torch.modules.discrete_cnn_module import DiscreteCNNModule
from garage.torch.modules.discrete_dueling_cnn_module import (
    DiscreteDuelingCNNModule)
# yapf: enable

__all__ = [
    'CategoricalCNNModule',
    'CNNModule',
    'DiscreteCNNModule',
    'DiscreteDuelingCNNModule',
    'MLPModule',
    'MultiHeadedMLPModule',
    'GaussianMLPModule',
    'GaussianMLPIndependentStdModule',
    'GaussianMLPTwoHeadedModule',
]
