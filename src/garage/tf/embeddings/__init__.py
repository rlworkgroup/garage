"""Embeddings."""
from garage.tf.embeddings.base import Embedding, EmbeddingSpec
from garage.tf.embeddings.base import StochasticEmbedding
from garage.tf.embeddings.gaussian_mlp_embedding import GaussianMLPEmbedding

__all__ = [
    'Embedding', 'EmbeddingSpec', 'StochasticEmbedding', 'GaussianMLPEmbedding'
]
