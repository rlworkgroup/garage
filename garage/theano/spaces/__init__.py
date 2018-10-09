"""Theano extension of garage.spaces."""

from garage.theano.spaces.box import Box
from garage.theano.spaces.dict import Dict
from garage.theano.spaces.discrete import Discrete
from garage.theano.spaces.tuple import Tuple

__all__ = ["Box", "Dict", "Discrete", "Tuple"]
