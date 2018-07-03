"""TensorFlow extension of garage.spaces."""

from garage.spaces.tf.box import Box
from garage.spaces.tf.discrete import Discrete
from garage.spaces.tf.product import Product

__all__ = ["Box", "Discrete", "Product"]
