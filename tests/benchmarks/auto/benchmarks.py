"""Run automatic benchmarks."""
from baselines.bench import benchmarks

from tests.benchmarks.auto.experiments import ddpg_garage_tf
from tests.benchmarks.auto.experiments import ppo_garage_pytorch
from tests.benchmarks.auto.experiments import ppo_garage_tf
from tests.benchmarks.auto.experiments import td3_garage_tf
from tests.benchmarks.auto.experiments import trpo_garage_pytorch
from tests.benchmarks.auto.experiments import trpo_garage_tf
from tests.benchmarks.auto.experiments import vpg_garage_pytorch
from tests.benchmarks.auto.experiments import vpg_garage_tf
from tests.benchmarks.auto.helper import iterate_experiments

_td3tasks = [
    task for task in benchmarks.get_benchmark('Mujoco1M')['tasks']
    if task['env_id'] != 'Reacher-v2'
]


def auto_benchmarks():
    """Run automatic benchmarks."""
    # garage-TensorFlow-DDPG
    for env_id, seed, log_dir in iterate_experiments(ddpg_garage_tf):
        ddpg_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)

    # garage-TensorFlow-PPO
    for env_id, seed, log_dir in iterate_experiments(ppo_garage_tf):
        ppo_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)

    # garage-PyTorch-PPO
    for env_id, seed, log_dir in iterate_experiments(ppo_garage_pytorch):
        ppo_garage_pytorch(dict(log_dir=log_dir), env_id=env_id, seed=seed)

    # garage-TensorFlow-TD3
    for env_id, seed, log_dir in iterate_experiments(td3_garage_tf, _td3tasks):
        td3_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)

    # garage-TensorFlow-TRPO
    for env_id, seed, log_dir in iterate_experiments(trpo_garage_tf):
        trpo_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)

    # garage-PyTorch-TRPO
    for env_id, seed, log_dir in iterate_experiments(trpo_garage_pytorch):
        trpo_garage_pytorch(dict(log_dir=log_dir), env_id=env_id, seed=seed)

    # garage-TensorFlow-VPG
    for env_id, seed, log_dir in iterate_experiments(vpg_garage_tf):
        vpg_garage_tf(dict(log_dir=log_dir), env_id=env_id, seed=seed)

    # garage-PyTorch-VPG
    for env_id, seed, log_dir in iterate_experiments(vpg_garage_pytorch):
        vpg_garage_pytorch(dict(log_dir=log_dir), env_id=env_id, seed=seed)
