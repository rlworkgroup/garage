# garage benchmarking

This guide explains how to write garage benchmark scripts.

### How to write a benchmark script?
**0. Install `garage_benchmarks`**

`cd benchmarks && pip install -e .`

**1. Define an experiment function**

Use `@wrap_experiment` to define your experiment. `ctxt` is used for `@wrap_experiment`.
The second and third parameters of the function should be `env_id` and `seed`.
You should give your function a good name because it is used as the label name when it comes to plotting.
See more [examples](https://github.com/rlworkgroup/garage/tree/master/benchmarks/src/garage_benchmarks/experiments).
```py
@wrap_experiment
def trpo_garage_pytorch(ctxt, env_id, seed):
    ...
```

**2. Define an execution function using `@benchmark` and `iterate_experiments()`**
```py
@benchmark
def your_benchmarks():
    iterate_experiments(trpo_garage_pytorch, MuJoCo1M_ENV_SET)
    iterate_experiments(trpo_garage_tf, MuJoCo1M_ENV_SET)
```
For the above example, the results will be saved under `/data/local/benchmarks/your_benchmarks`.
Plotting is enabled by default and figures will be stored in `/plot`.
You can explicitly disable it by defining `@benchmark(plot=False)`.

**3. Trigger your benchmark**

Under `garage/benchmarks` do:

`garage_benchmark run your_benchmarks`


### Environment sets
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
 'MemorizeDigits-v0', 'CubeCrash-v0', ^'Acrobot-v1', ^'MountainCar-v0', ^'CartPole-v1', ^'LunarLander-v2']
```
^ Using the wrappers [PixelObservationWrapper](https://github.com/openai/gym/blob/master/gym/wrappers/pixel_observation.py) and [FrameStack](https://github.com/openai/gym/blob/master/gym/wrappers/frame_stack.py) (n=4)
```
ML_ENV_SET: [
 'ML1-push-v1', 'ML1-reach-v1', 'ML1-pick-place-v1', 'ML10', 'ML45']
```
```
MT_ENV_SET: [
 'ML1-push-v1', 'ML1-reach-v1', 'ML1-pick-place-v1', 'MT10', 'MT50']
```
See [Meta-World](https://github.com/rlworkgroup/metaworld) for ML1/ML10/ML45/MT10/MT50.
```
STATE_ENV_SET: [
 'LunarLander-v2', 'CartPole-v1', 'Assault-ramDeterministic-v4', 'Breakout-ramDeterministic-v4', 'ChopperCommand-ramDeterministic-v4', 'Tutankham-ramDeterministic-v4']
```
