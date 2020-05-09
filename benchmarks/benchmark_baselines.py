"""Benchmarking for baselines."""
import random

from benchmarks import benchmark, iterate_experiments
from benchmarks.experiments.baselines import continuous_mlp_baseline_tf
from benchmarks.experiments.baselines import gaussian_cnn_baseline_tf
from benchmarks.experiments.baselines import gaussian_mlp_baseline_tf
from benchmarks.parameters import MuJoCo1M_ENV_SET, PIXEL_ENV_SET

_seeds = random.sample(range(100), 3)


@benchmark
def ppo_continuous_mlp_baseline_tf_benchmarks():
    """Run experiments for Continuous MLP Baseline TF benchmarking."""
    iterate_experiments(continuous_mlp_baseline_tf,
                        MuJoCo1M_ENV_SET,
                        seeds=_seeds)


@benchmark
def ppo_gaussian_cnn_baseline_tf_benchmarks():
    """Run experiments for Gaussian CNN Baseline TF benchmarking."""
    iterate_experiments(gaussian_cnn_baseline_tf, PIXEL_ENV_SET, seeds=_seeds)


@benchmark
def ppo_gaussian_mlp_baseline_tf_benchmarks():
    """Run experiments for Gaussian MLP Baseline TF benchmarking."""
    iterate_experiments(gaussian_mlp_baseline_tf,
                        MuJoCo1M_ENV_SET,
                        seeds=_seeds)
