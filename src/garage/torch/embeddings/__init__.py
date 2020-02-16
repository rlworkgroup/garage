"""PyTorch embedding modules for meta-learning algorithms."""
from garage.torch.embeddings.mlp_encoder import MLPEncoder
from garage.torch.embeddings.recurrent_encoder import RecurrentEncoder

__all__ = ['MLPEncoder', 'RecurrentEncoder']
