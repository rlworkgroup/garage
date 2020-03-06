"""PyTorch embedding modules for meta-learning algorithms."""

from garage.torch.embeddings.context_encoder import ContextEncoder
from garage.torch.embeddings.mlp_encoder import MLPEncoder

__all__ = [
    'ContextEncoder',
    'MLPEncoder',
]
