"""Run automatic benchmarking."""
from baselines.bench import benchmarks

from tests.benchmarks.auto.experiments import ddpg_garage_tf
from tests.benchmarks.auto.experiments import ppo_garage_pytorch
from tests.benchmarks.auto.experiments import ppo_garage_tf
from tests.benchmarks.auto.experiments import td3_garage_tf
from tests.benchmarks.auto.experiments import trpo_garage_pytorch
from tests.benchmarks.auto.experiments import trpo_garage_tf
from tests.benchmarks.auto.experiments import vpg_garage_pytorch
from tests.benchmarks.auto.experiments import vpg_garage_tf
from tests.benchmarks.auto.helper import benchmark, iterate_experiments

_td3_env_ids = [
    task['env_id'] for task in benchmarks.get_benchmark('Mujoco1M')['tasks']
    if task['env_id'] != 'Reacher-v2'
]


@benchmark(plot=False, auto=True)
def auto_benchmarks():
    """Run experiments for automatic benchmarking."""
    iterate_experiments(ddpg_garage_tf)
    iterate_experiments(ppo_garage_tf)
    iterate_experiments(ppo_garage_pytorch)
    iterate_experiments(td3_garage_tf, _td3_env_ids)
    iterate_experiments(trpo_garage_tf)
    iterate_experiments(trpo_garage_pytorch)
    iterate_experiments(vpg_garage_tf)
    iterate_experiments(vpg_garage_pytorch)
