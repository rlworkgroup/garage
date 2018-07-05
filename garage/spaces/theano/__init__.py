"""Theano extension of garage.spaces."""

from garage.spaces.theano.box import Box
from garage.spaces.theano.discrete import Discrete
from garage.spaces.theano.product import Product

__all__ = ["Box", "Discrete", "Product"]
