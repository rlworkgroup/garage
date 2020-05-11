"""Benchmarking experiments for algorithms."""
from garage_benchmarks.experiments.algos.ddpg_garage_tf import ddpg_garage_tf
from garage_benchmarks.experiments.algos.her_garage_tf import her_garage_tf
from garage_benchmarks.experiments.algos.ppo_garage_pytorch import (
    ppo_garage_pytorch)
from garage_benchmarks.experiments.algos.ppo_garage_tf import ppo_garage_tf
from garage_benchmarks.experiments.algos.td3_garage_tf import td3_garage_tf
from garage_benchmarks.experiments.algos.trpo_garage_pytorch import (
    trpo_garage_pytorch)
from garage_benchmarks.experiments.algos.trpo_garage_tf import trpo_garage_tf
from garage_benchmarks.experiments.algos.vpg_garage_pytorch import (
    vpg_garage_pytorch)
from garage_benchmarks.experiments.algos.vpg_garage_tf import vpg_garage_tf

__all__ = [
    'ddpg_garage_tf', 'her_garage_tf', 'ppo_garage_pytorch', 'ppo_garage_tf',
    'td3_garage_tf', 'trpo_garage_pytorch', 'trpo_garage_tf',
    'vpg_garage_pytorch', 'vpg_garage_tf'
]
