"""Baseline estimators for TensorFlow-based algorithms"""
from garage.tf.baselines.continuous_mlp_baseline import ContinuousMLPBaseline
from garage.tf.baselines.continuous_mlp_baseline_with_model import (
    ContinuousMLPBaselineWithModel)
from garage.tf.baselines.gaussian_conv_baseline import GaussianConvBaseline
from garage.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from garage.tf.baselines.gaussian_mlp_baseline_with_model import (
    GaussianMLPBaselineWithModel)

__all__ = [
    'ContinuousMLPBaseline',
    'ContinuousMLPBaselineWithModel',
    'GaussianConvBaseline',
    'GaussianMLPBaseline',
    'GaussianMLPBaselineWithModel',
]
