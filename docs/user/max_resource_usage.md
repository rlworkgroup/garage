# Maximize resource usage

To accelerate running an experiment, we want to maximize the usage of resources
(CPUs or GPUs).

## Maximize CPU utilization

The most compute-intensive part in training a policy is sampling. Garage uses
[`Worker`](sampling.html#worker) to abstract a CPU to perform a rollout and uses
[`Sampler`](sampling.html#sampler) to manage them. To maximize CPU utilization,
we only need to choose proper `Sampler` and `Worker` when setting up
`LocalRunner`.

In the following tests, we will use [`examples/torch/trpo_pendulum.py`](https://github.com/rlworkgroup/garage/blob/master/examples/torch/trpo_pendulum.py)
as example experiment file.

### `LocalSampler` and `DefaultWorker`

First, we use the most basic config, `LocalSampler` and `DefaultWorker`. The
sampler will run workers in the same process.

```py
from garage.sampler import LocalSampler, DefaultWorker

    ...
    runner.setup(algo, env, sampler_cls=LocalSampler, worker_class=DefaultWorker)
    ...
```

With `top` command, in my environment (4 cores), we can see the CPU usage is
about 300%.

```sh
...
  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
14610 ruofu     20   0 4555092 558680 268692 R 301.0  6.9   6:06.37 python examples
...
```

### Use `VecWorker`

To reduce the overhead in sampling, we use `VecWorker` to run multiple
environments in one step.

```py
from garage.sampler import LocalSampler, VecWorker

    ...
    runner.setup(algo, env, sampler_cls=LocalSampler, worker_class=VecWorker)
    ...
```

And the CPU usage is about 350%.

```sh
...
  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
11902 ruofu     20   0 4696076 709944 268424 R 351.2  8.7   2:56.95 python examples
...
```

### Use `RaySampler`

Though the CPU usage is increased, it still run in one process. We can
parallelize sampling across CPUs using `RaySampler` to maximize the CPU usage:

```py
from garage.sampler import RaySampler, VecWorker

    ...
    runner.setup(algo, env, sampler_cls=RaySampler, worker_class=VecWorker)
    ...
```

From `top` command, we get:

```sh
...
  PID USER      PR  NI    VIRT    RES    SHR S  %CPU %MEM     TIME+ COMMAND
12652 ruofu     20   0 7471988 570096 273596 S 380.7  7.0   7:12.44 python examples
12714 ruofu     20   0 6303456 554708 264884 S  24.3  6.8   0:30.03 ray::SamplerWor
12720 ruofu     20   0 6303516 553376 265004 S  23.9  6.8   0:29.79 ray::SamplerWor
12715 ruofu     20   0 6303456 555516 265112 S  22.9  6.8   0:29.34 ray::SamplerWor
12721 ruofu     20   0 6303516 556660 264656 S  22.9  6.8   0:29.52 ray::SamplerWor
...
```

We can see that there are 4 `ray::SamplerWor` processes running, which are
parallelled workers for sampling.

## Use GPU

For algorithms with PyTorch or TensorFlow, we can use GPU to train policies.

### PyTorch

In Garage, PyTorch use CPU mode by default. To enable GPU mode, use
`set_gpu_mode()` function after declaring an algorithm in the experiment
function. For example:

```py
import torch
from garage.torch import set_gpu_mode

@wrap_experiment
def trpo_pendulum(ctxt=None, seed=1):

    ...
    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                max_path_length=100,
                discount=0.99,
                center_adv=False)

    # enable GPU
    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    algo.to()

    runner.setup(algo, env)
    runner.train(n_epochs=1000, batch_size=1024)
```

### TensorFlow

Different from PyTorch, TensorFlow will use a GPU if one is available by
default. To disable it, you can execute the following command before run the
experiment launcher.

```sh
export CUDA_VISIBLE_DEVICES=-1  # CPU only
```

You can see more information about using GPU [here](experiments).

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*
