# Automated Benchmarking
[Benchmark repo](https://github.com/rlworkgroup/benchmarks)

We have an automated system that will run any benchmark file named "test_benchmark_*.py" in this folder /tests/benchmarks/.

The results are displayed [here](https://rlworkgroup.github.io/benchmarks/)

# Environments
| Algorithm | Observation | Action | Environment Set |
| --- | --- | --- | --- |
| On-policy | Pixel | Discrete | *PIXEL_ENV_SET |
| Off-policy | Pixel | Discrete | Atari1M |
| Meta-RL | Non-Pixel | Discrete | *ML_ENV_SET |
| MultiTask-RL | Non-Pixel | Discrete | *MT_ENV_SET |
| ALL | Non-Pixel | Discrete | *NON_PIXEL_ENV_SET |
| ALL | Non-Pixel | Continuous | MuJoCo1M |

```
PIXEL_ENV_SET: [
    'MemorizeDigits-v0',
    'CubeCrash-v0',
    ^'Acrobot-v1',
    ^'MountainCar-v0',
    ^'CartPole-v1',
    ^'LunarLander-v2'
]
```
^ Using the wrappers [PixelObservationWrapper](https://github.com/openai/gym/blob/master/gym/wrappers/pixel_observation.py) and [FrameStack](https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py) (n=4)
```
ML_ENV_SET: [
    'ML1-push-v1',
    'ML1-reach-v1',
    'ML1-pick-place-v1',
    'ML10',
    'ML45'
]
```
```
MT_ENV_SET: [
    'ML1-push-v1',
    'ML1-reach-v1',
    'ML1-pick-place-v1',
    'MT10',
    'MT50'
]
```
See (Meta-Wrold)[https://github.com/rlworkgroup/metaworld] for ML1/ML10/ML45/MT10/MT50.
```
STATE_ENV_SET: [
    'LunarLander-v2',
    'CartPole-v1',
    'Assault-ramDeterministic-v4',
    'Breakout-ramDeterministic-v4',
    'ChopperCommand-ramDeterministic-v4',
    'Tutankham-ramDeterministic-v4'
]
```
# Directions
Follow these directions to output your benchmark results as a json compatible with the display.
```python
from tests.helpers import create_json, write_file, plot

result_json = {}
for t in tasks:
    result_json[task] = create_json()

write_file(result_json, 'Algo name')
```

# Example

```python
@pytest.mark.huge
def test_benchmark_ddpg(self):
    #
    #
    #
    result_json = {}
    for task in mujoco1m['tasks']:
        #
        #
        #
        plot()

        result_json[env_id] = create_json(
            b_csvs=baselines_csvs,
            g_csvs=garage_csvs,
            seeds=seeds,
            trails=task['trials'],
            g_x='Epoch',
            g_y='AverageReturn',
            b_x='total/epochs',s
            b_y='rollout/return',
            factor_g=TimeStepsConversionFactorGarage,
            factor_b=TimeStepsConversionFactorBenchmark)

    write_file(result_json, 'DDPG')
```

# Json Schema
For each benchmark script, the following json schema is produced.

```json
{
    "algo1": {
        "task_1": {
            {
                "seed": 1,
                "trail_1": {
                    "garage": {
                        "time_steps": [...],
                        "average_return" : [...]
                    },
                    "benchmark":{
                        "time_steps": [...],
                        "average_return" : [...]
                    }
                },
            },
            {
                "seed":2,
                "trail_2":{...}
            },
            {
                "seed":3,
                "trail_3":{...}
            }
        },
        "task_2":{...},
        "task_3":{...}
        }
    }
```

The collection of these jsons are consolidated by benchmarks/compile_file.py, which combines them in the following format:
```json
{
    "timestamp": "20xx-xx-xxTxx:xx:xxZ",
    "tasks": ["task_1", "task_2", "task_3",...],
    "algos": {
        "algo1":{...},
        "algo2":{...},
        "algo3":{...}
    }
}
```
