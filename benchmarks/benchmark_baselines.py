"""Benchmarking for baselines."""
import random

from benchmarks import benchmark, iterate_experiments
from benchmarks.experiments.baselines import continuous_mlp_baseline_tf
from benchmarks.experiments.baselines import gaussian_cnn_baseline_tf
from benchmarks.experiments.baselines import gaussian_mlp_baseline_tf


@benchmark
def ppo_continuous_mlp_baseline_tf_benchmarks():
    """Run experiments for Continuous MLP Baseline TF benchmarking."""
    seeds = random.sample(range(100), 3)
    iterate_experiments(continuous_mlp_baseline_tf, seeds=seeds)


@benchmark
def ppo_gaussian_cnn_baseline_tf_benchmarks():
    """Run experiments for Gaussian CNN Baseline TF benchmarking."""
    env_ids = ['CubeCrash-v0', 'MemorizeDigits-v0']
    seeds = random.sample(range(100), 3)
    iterate_experiments(gaussian_cnn_baseline_tf, env_ids=env_ids, seeds=seeds)


@benchmark
def ppo_gaussian_mlp_baseline_tf_benchmarks():
    """Run experiments for Gaussian MLP Baseline TF benchmarking."""
    env_ids = [
        'HalfCheetah-v2', 'Reacher-v2', 'Walker2d-v2', 'Hopper-v2',
        'Swimmer-v2', 'InvertedPendulum-v2', 'InvertedDoublePendulum-v2'
    ]
    seeds = random.sample(range(100), 3)
    iterate_experiments(gaussian_mlp_baseline_tf, env_ids=env_ids, seeds=seeds)
