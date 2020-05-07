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
from benchmarks.parameters import MuJoCo1M_ENV_SET


@benchmark
def ddpg_benchmarks():
    """Run experiments for DDPG benchmarking."""
    iterate_experiments(ddpg_garage_tf, MuJoCo1M_ENV_SET)


@benchmark
def her_benchmarks():
    """Run experiments for HER benchmarking."""
    her_end_ids = [
        task['env_id'] for task in benchmarks.get_benchmark('Fetch1M')['tasks']
    ]

    iterate_experiments(her_garage_tf, her_end_ids)


@benchmark
def ppo_benchmarks():
    """Run experiments for PPO benchmarking."""
    iterate_experiments(ppo_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(ppo_garage_tf, MuJoCo1M_ENV_SET)


@benchmark
def td3_benchmarks():
    """Run experiments for TD3 benchmarking."""
    td3_env_ids = [
        env_id for env_id in MuJoCo1M_ENV_SET if env_id != 'Reacher-v2'
    ]

    iterate_experiments(td3_garage_tf, td3_env_ids)


@benchmark
def trpo_benchmarks():
    """Run experiments for TRPO benchmarking."""
    iterate_experiments(trpo_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(trpo_garage_tf, MuJoCo1M_ENV_SET)


@benchmark
def vpg_benchmarks():
    """Run experiments for VPG benchmarking."""
    iterate_experiments(vpg_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(vpg_garage_tf, MuJoCo1M_ENV_SET)
