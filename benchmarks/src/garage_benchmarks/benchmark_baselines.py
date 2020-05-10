"""Benchmarking for baselines."""
import random

from garage_benchmarks.experiments.baselines import continuous_mlp_baseline
from garage_benchmarks.experiments.baselines import gaussian_cnn_baseline
from garage_benchmarks.experiments.baselines import gaussian_mlp_baseline
from garage_benchmarks.helper import benchmark, iterate_experiments
from garage_benchmarks.parameters import MuJoCo1M_ENV_SET, PIXEL_ENV_SET

_seeds = random.sample(range(100), 3)


@benchmark
def continuous_mlp_baseline_tf_ppo_benchmarks():
    """Run benchmarking experiments for Continuous MLP Baseline on TF-PPO."""
    iterate_experiments(continuous_mlp_baseline,
                        MuJoCo1M_ENV_SET,
                        seeds=_seeds)


@benchmark
def gaussian_cnn_baseline_tf_ppo_benchmarks():
    """Run benchmarking experiments for Gaussian CNN Baseline on TF-PPO."""
    iterate_experiments(gaussian_cnn_baseline, PIXEL_ENV_SET, seeds=_seeds)


@benchmark
def gaussian_mlp_baseline_tf_ppo_benchmarks():
    """Run benchmarking experiments for Gaussian MLP Baseline on TF-PPO."""
    iterate_experiments(gaussian_mlp_baseline, MuJoCo1M_ENV_SET, seeds=_seeds)
