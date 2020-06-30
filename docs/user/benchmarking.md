# Benchmarking

This guide explains how to write and run garage benchmark scripts.

## Prerequisites

Firstly, you should have installed garage. If not, you can see instructions in
the [latest documentation](installation) to install garage.

## Install `garage-benchmarks`

`cd garage/benchmarks && pip install -e .`

## Write garage benchmark scripts

### Define an experiment function

Use `@wrap_experiment` to define your experiment. `ctxt` is used for
`@wrap_experiment`.
The second and third parameters of the function should be `env_id` and `seed`.
You should give your function a good name because it is used as the label name
when it comes to plotting. See more [examples](https://github.com/rlworkgroup/garage/tree/master/benchmarks/src/garage_benchmarks/experiments).
You can see detailed documentations of defining experiments [here](experiments).

```py
@wrap_experiment
def trpo_garage_pytorch(ctxt, env_id, seed):
    ...
```

### Define an execution function using `@benchmark` and `iterate_experiments()`

Plotting is enabled by default. You can explicitly disable it by defining
`@benchmark(plot=False)`.

```py
@benchmark
def your_benchmarks():
    iterate_experiments(trpo_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(trpo_garage_tf, MuJoCo1M_ENV_SET)
```

The execution function should be placed in one of the following files:

```bash
garage/benchmarks/src/garage_benchmarks
                      ├── benchmark_algos.py
                      ├── benchmark_auto.py
                      ├── benchmark_baselines.py
                      ├── benchmark_policies.py
                      └── benchmark_q_functions.py
```

## Run garage benchmarks

To list all the benchmarks, do:

`garage_benchmark list`

To run a benchmark, do:

`garage_benchmark run your_benchmark`

For the above example, the results will be saved in `./data/local/benchmarks/your_benchmark/`.

## Environment sets

```eval_rst
+---------------+-------------+------------+--------------------+
| Algorithm     | Observation | Action     | Environment Set    |
+===============+=============+============+====================+
| On-policy     | Pixel       | Discrete   | *PIXEL_ENV_SET     |
+---------------+-------------+------------+--------------------+
| Off-policy    | Pixel       | Discrete   | Atari1M            |
+---------------+-------------+------------+--------------------+
| Meta-RL       | Non-Pixel   | Discrete   | *ML_ENV_SET        |
+---------------+-------------+------------+--------------------+
| Multi-Task RL | Non-Pixel   | Discrete   | *MT_ENV_SET        |
+---------------+-------------+------------+--------------------+
| ALL           | Non-Pixel   | Discrete   | *NON_PIXEL_ENV_SET |
+---------------+-------------+------------+--------------------+
| ALL           | Non-Pixel   | Continuous | MuJoCo-1M          |
+---------------+-------------+------------+--------------------+
```

```py
PIXEL_ENV_SET = [
 'MemorizeDigits-v0', 'CubeCrash-v0', ^'Acrobot-v1', ^'MountainCar-v0',
 ^'CartPole-v1', ^'LunarLander-v2']
```

^ Using the wrappers [PixelObservationWrapper](https://github.com/openai/gym/blob/master/gym/wrappers/pixel_observation.py)
and [FrameStack](https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py)
(n=4)

```py
ML_ENV_SET = [
 'ML1-push-v1', 'ML1-reach-v1', 'ML1-pick-place-v1', 'ML10', 'ML45']
```

```py
MT_ENV_SET = [
 'ML1-push-v1', 'ML1-reach-v1', 'ML1-pick-place-v1', 'MT10', 'MT50']
```

See [Meta-World](https://github.com/rlworkgroup/metaworld) for ML1/ML10/ML45/MT10/MT50.

```py
STATE_ENV_SET = [
 'LunarLander-v2', 'CartPole-v1', 'Assault-ramDeterministic-v4',
 'Breakout-ramDeterministic-v4', 'ChopperCommand-ramDeterministic-v4',
 'Tutankham-ramDeterministic-v4']
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*
