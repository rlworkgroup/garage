"""Benchmarking for algorithms."""
from baselines.bench import benchmarks
from benchmarks import benchmark, iterate_experiments
from benchmarks.experiments.algos import ddpg_garage_tf
from benchmarks.experiments.algos import her_garage_tf
from benchmarks.experiments.algos import ppo_garage_pytorch
from benchmarks.experiments.algos import ppo_garage_tf
from benchmarks.experiments.algos import td3_garage_tf
from benchmarks.experiments.algos import trpo_garage_pytorch
from benchmarks.experiments.algos import trpo_garage_tf
from benchmarks.experiments.algos import vpg_garage_pytorch
from benchmarks.experiments.algos import vpg_garage_tf


@benchmark
def ddpg_benchmarks():
    """Run experiments for DDPG benchmarking."""
    iterate_experiments(ddpg_garage_tf)


@benchmark
def her_benchmarks():
    """Run experiments for HER benchmarking."""
    her_end_ids = [
        task['env_id'] for task in benchmarks.get_benchmark('Fetch1M')['tasks']
    ]

    iterate_experiments(her_garage_tf, env_ids=her_end_ids)


@benchmark
def ppo_benchmarks():
    """Run experiments for PPO benchmarking."""
    iterate_experiments(ppo_garage_pytorch)
    iterate_experiments(ppo_garage_tf)


@benchmark
def td3_benchmarks():
    """Run experiments for TD3 benchmarking."""
    td3_env_ids = [
        task['env_id']
        for task in benchmarks.get_benchmark('Mujoco1M')['tasks']
        if task['env_id'] != 'Reacher-v2'
    ]

    iterate_experiments(td3_garage_tf, env_ids=td3_env_ids)


@benchmark
def trpo_benchmarks():
    """Run experiments for TRPO benchmarking."""
    iterate_experiments(trpo_garage_pytorch)
    iterate_experiments(trpo_garage_tf)


@benchmark
def vpg_benchmarks():
    """Run experiments for VPG benchmarking."""
    iterate_experiments(vpg_garage_pytorch)
    iterate_experiments(vpg_garage_tf)
