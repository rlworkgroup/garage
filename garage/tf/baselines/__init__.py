"""Baseline estimators for TensorFlow-based algorithms"""
from garage.tf.baselines.deterministic_mlp_baseline import (
    DeterministicMLPBaseline)
from garage.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

__all__ = ["DeterministicMLPBaseline", "GaussianMLPBaseline"]
