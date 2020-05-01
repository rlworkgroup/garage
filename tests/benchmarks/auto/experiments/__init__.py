"""Automatic benchmarking experiments."""
from tests.benchmarks.auto.experiments.ddpg_garage_tf import ddpg_garage_tf
from tests.benchmarks.auto.experiments.ppo_garage_pytorch import (
    ppo_garage_pytorch)
from tests.benchmarks.auto.experiments.ppo_garage_tf import ppo_garage_tf
from tests.benchmarks.auto.experiments.td3_garage_tf import td3_garage_tf
from tests.benchmarks.auto.experiments.trpo_garage_pytorch import (
    trpo_garage_pytorch)
from tests.benchmarks.auto.experiments.trpo_garage_tf import trpo_garage_tf
from tests.benchmarks.auto.experiments.vpg_garage_pytorch import (
    vpg_garage_pytorch)
from tests.benchmarks.auto.experiments.vpg_garage_tf import vpg_garage_tf

__all__ = [
    'ddpg_garage_tf', 'ppo_garage_pytorch', 'ppo_garage_tf', 'td3_garage_tf',
    'trpo_garage_pytorch', 'trpo_garage_tf', 'vpg_garage_pytorch',
    'vpg_garage_tf'
]
