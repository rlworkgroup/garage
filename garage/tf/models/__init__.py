"""Tensorflow models."""

from garage.tf.models.base import Model
from garage.tf.models.gaussian_mlp_model import GaussianMLPModel

__all__ = ["Model", "GaussianMLPModel"]
