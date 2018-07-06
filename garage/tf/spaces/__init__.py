"""TensorFlow extension of garage.spaces."""

from garage.tf.spaces.box import Box
from garage.tf.spaces.discrete import Discrete
from garage.tf.spaces.product import Product

__all__ = ["Box", "Discrete", "Product"]
