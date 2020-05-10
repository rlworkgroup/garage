"""Benchmarking experiments for baselines."""
from garage_benchmarks.experiments.policies.categorical_cnn_policy import (
    categorical_cnn_policy)
from garage_benchmarks.experiments.policies.categorical_gru_policy import (
    categorical_gru_policy)
from garage_benchmarks.experiments.policies.categorical_lstm_policy import (
    categorical_lstm_policy)
from garage_benchmarks.experiments.policies.categorical_mlp_policy import (
    categorical_mlp_policy)
from garage_benchmarks.experiments.policies.continuous_mlp_policy import (
    continuous_mlp_policy)
from garage_benchmarks.experiments.policies.gaussian_gru_policy import (
    gaussian_gru_policy)
from garage_benchmarks.experiments.policies.gaussian_lstm_policy import (
    gaussian_lstm_policy)
from garage_benchmarks.experiments.policies.gaussian_mlp_policy import (
    gaussian_mlp_policy)

__all__ = [
    'categorical_cnn_policy', 'categorical_gru_policy',
    'categorical_lstm_policy', 'categorical_mlp_policy',
    'continuous_mlp_policy', 'gaussian_gru_policy', 'gaussian_lstm_policy',
    'gaussian_mlp_policy'
]
