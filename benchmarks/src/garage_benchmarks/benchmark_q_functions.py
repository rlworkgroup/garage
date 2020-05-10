"""Benchmarking for q-functions."""
import random

from garage_benchmarks.experiments.q_functions import continuous_mlp_q_function
from garage_benchmarks.helper import benchmark, iterate_experiments
from garage_benchmarks.parameters import MuJoCo1M_ENV_SET

_seeds = random.sample(range(100), 5)


@benchmark
def continuous_mlp_q_function_tf_ddpg_benchmarks():
    """Run benchmarking experiments for Continuous MLP QFunction on TF-DDPG."""
    iterate_experiments(continuous_mlp_q_function,
                        MuJoCo1M_ENV_SET,
                        seeds=_seeds)
