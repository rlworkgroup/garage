"""Baseline estimators for TensorFlow-based algorithms."""
from garage.tf.baselines.continuous_mlp_baseline import ContinuousMLPBaseline
from garage.tf.baselines.continuous_mlp_baseline_with_model import (
    ContinuousMLPBaselineWithModel)
from garage.tf.baselines.gaussian_cnn_baseline_with_model import (
    GaussianCNNBaselineWithModel)
from garage.tf.baselines.gaussian_conv_baseline import GaussianConvBaseline
from garage.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline

__all__ = [
    'ContinuousMLPBaseline',
    'ContinuousMLPBaselineWithModel',
    'GaussianConvBaseline',
    'GaussianCNNBaselineWithModel',
    'GaussianMLPBaseline',
]
