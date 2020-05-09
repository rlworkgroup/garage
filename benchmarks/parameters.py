"""Global parameters for benchmarking."""
from baselines.bench import benchmarks

MuJoCo1M_ENV_SET = [
    task['env_id'] for task in benchmarks.get_benchmark('Mujoco1M')['tasks']
]

PIXEL_ENV_SET = ['CubeCrash-v0', 'MemorizeDigits-v0']
