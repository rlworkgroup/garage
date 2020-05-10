"""Benchmarking for policies."""
import random

from garage_benchmarks.experiments.policies import categorical_cnn_policy
from garage_benchmarks.experiments.policies import categorical_gru_policy
from garage_benchmarks.experiments.policies import categorical_lstm_policy
from garage_benchmarks.experiments.policies import categorical_mlp_policy
from garage_benchmarks.experiments.policies import continuous_mlp_policy
from garage_benchmarks.experiments.policies import gaussian_gru_policy
from garage_benchmarks.experiments.policies import gaussian_lstm_policy
from garage_benchmarks.experiments.policies import gaussian_mlp_policy
from garage_benchmarks.helper import benchmark, iterate_experiments
from garage_benchmarks.parameters import (MuJoCo1M_ENV_SET, PIXEL_ENV_SET,
                                          STATE_ENV_SET)

_seeds = random.sample(range(100), 3)


@benchmark
def categorical_cnn_policy_tf_ppo_benchmarks():
    """Run benchmarking experiments for Categorical CNN Policy on TF-PPO."""
    iterate_experiments(categorical_cnn_policy, PIXEL_ENV_SET, seeds=_seeds)


@benchmark
def categorical_gru_policy_tf_ppo_benchmarks():
    """Run benchmarking experiments for Categorical GRU Policy on TF-PPO."""
    iterate_experiments(categorical_gru_policy, STATE_ENV_SET, seeds=_seeds)


@benchmark
def categorical_lstm_policy_tf_ppo_benchmarks():
    """Run benchmarking experiments for Categorical LSTM Policy on TF-PPO."""
    iterate_experiments(categorical_lstm_policy, STATE_ENV_SET, seeds=_seeds)


@benchmark
def categorical_mlp_policy_tf_ppo_benchmarks():
    """Run benchmarking experiments for Categorical MLP Policy on TF-PPO."""
    iterate_experiments(categorical_mlp_policy, STATE_ENV_SET, seeds=_seeds)


@benchmark
def continuous_mlp_policy_tf_ddpg_benchmarks():
    """Run benchmarking experiments for Continuous MLP Policy on TF-DDPG."""
    seeds = random.sample(range(100), 5)
    iterate_experiments(continuous_mlp_policy, MuJoCo1M_ENV_SET, seeds=seeds)


@benchmark
def gaussian_gru_policy_tf_ppo_benchmarks():
    """Run benchmarking experiments for Gaussian GRU Policy on TF-PPO."""
    iterate_experiments(gaussian_gru_policy, MuJoCo1M_ENV_SET, seeds=_seeds)


@benchmark
def gaussian_lstm_policy_tf_ppo_benchmarks():
    """Run benchmarking experiments for Gaussian LSTM Policy on TF-PPO."""
    iterate_experiments(gaussian_lstm_policy, MuJoCo1M_ENV_SET, seeds=_seeds)


@benchmark
def gaussian_mlp_policy_tf_ppo_benchmarks():
    """Run benchmarking experiments for Gaussian MLP Policy on TF-PPO."""
    iterate_experiments(gaussian_mlp_policy, MuJoCo1M_ENV_SET, seeds=_seeds)
