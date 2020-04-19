"""Encoders in TensorFlow."""
# pylint: disable=abstract-method
from garage.np.embeddings import Encoder as BaseEncoder
from garage.np.embeddings import StochasticEncoder as BaseStochasticEncoder
from garage.tf.models import Module, StochasticModule


class Encoder(BaseEncoder, Module):
    """Base class for encoders in TensorFlow."""


class StochasticEncoder(BaseStochasticEncoder, StochasticModule):
    """Base class for stochastic encoders in TensorFlow."""
