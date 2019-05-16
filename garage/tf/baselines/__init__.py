"""Baseline estimators for TensorFlow-based algorithms"""
from garage.tf.baselines.continuous_mlp_baseline_with_model import (
    ContinuousMLPBaselineWithModel)
from garage.tf.baselines.deterministic_mlp_baseline import (
    DeterministicMLPBaseline)
from garage.tf.baselines.gaussian_conv_baseline import GaussianConvBaseline
from garage.tf.baselines.gaussian_mlp_baseline import GaussianMLPBaseline
from garage.tf.baselines.gaussian_mlp_baseline_with_model import (
    GaussianMLPBaselineWithModel)

__all__ = [
    'ContinuousMLPBaselineWithModel',
    'DeterministicMLPBaseline',
    'GaussianConvBaseline',
    'GaussianMLPBaseline',
    'GaussianMLPBaselineWithModel',
]
