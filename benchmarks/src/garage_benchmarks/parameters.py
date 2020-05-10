"""Global parameters for benchmarking."""
from baselines.bench import benchmarks

Fetch1M_ENV_SET = [
    task['env_id'] for task in benchmarks.get_benchmark('Fetch1M')['tasks']
]

MuJoCo1M_ENV_SET = [
    task['env_id'] for task in benchmarks.get_benchmark('Mujoco1M')['tasks']
]

PIXEL_ENV_SET = ['CubeCrash-v0', 'MemorizeDigits-v0']

STATE_ENV_SET = [
    'LunarLander-v2',
    'Assault-ramDeterministic-v4',
    'Breakout-ramDeterministic-v4',
    'ChopperCommand-ramDeterministic-v4',
    'Tutankham-ramDeterministic-v4',
    'CartPole-v1',
]
