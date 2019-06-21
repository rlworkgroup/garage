# Automated Benchmarking
[Benchmark repo](https://github.com/rlworkgroup/benchmarks)

We have an automated system that will run any benchmark file named "test_benchmark_*.py" in this folder /tests/benchmarks/.

The results are displayed [here](https://rlworkgroup.github.io/benchmarks/)

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
