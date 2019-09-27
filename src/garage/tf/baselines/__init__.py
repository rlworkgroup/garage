"""Baseline estimators for TensorFlow-based algorithms."""
from garage.tf.baselines.continuous_mlp_baseline import ContinuousMLPBaseline
from garage.tf.baselines.gaussian_cnn_baseline import GaussianCNNBaseline
from garage.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

__all__ = [
    'ContinuousMLPBaseline',
    'GaussianCNNBaseline',
    'GaussianMLPBaseline',
]
