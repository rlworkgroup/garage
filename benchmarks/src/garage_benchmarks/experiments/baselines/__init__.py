"""Benchmarking experiments for baselines."""
from garage_benchmarks.experiments.baselines.continuous_mlp_baseline import (
    continuous_mlp_baseline)
from garage_benchmarks.experiments.baselines.gaussian_cnn_baseline import (
    gaussian_cnn_baseline)
from garage_benchmarks.experiments.baselines.gaussian_mlp_baseline import (
    gaussian_mlp_baseline)

__all__ = [
    'continuous_mlp_baseline', 'gaussian_cnn_baseline', 'gaussian_mlp_baseline'
]
