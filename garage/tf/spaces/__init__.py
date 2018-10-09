"""TensorFlow extension of garage.spaces."""

from garage.tf.spaces.box import Box
from garage.tf.spaces.dict import Dict
from garage.tf.spaces.discrete import Discrete
from garage.tf.spaces.tuple import Tuple

__all__ = ["Box", "Dict", "Discrete", "Tuple"]
