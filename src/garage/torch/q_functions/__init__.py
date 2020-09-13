"""PyTorch Q-functions."""
from garage.torch.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction)
from garage.torch.q_functions.discrete_cnn_q_function import (
    DiscreteCNNQFunction)

__all__ = ['ContinuousMLPQFunction', 'DiscreteCNNQFunction']
