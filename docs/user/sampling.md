# Sampling

`LocalRunner` gets trajectories through Sampling to train the policy. In
Garage, [`LocalRunner`](https://github.com/rlworkgroup/garage/blob/a767dcb95988abc14165a2a320ee39932a1c85ca/src/garage/experiment/local_runner.py)
uses [`Sampler`](https://github.com/rlworkgroup/garage/blob/a767dcb95988abc14165a2a320ee39932a1c85ca/src/garage/sampler/sampler.py)
to perform sampling. `Sampler` manages [`Worker`](https://github.com/rlworkgroup/garage/blob/a767dcb95988abc14165a2a320ee39932a1c85ca/src/garage/sampler/worker.py#L5)s
and assign specific tasks to them, such as performing rollouts and update
agents and environments.

## Sampler

Sampler is responsible for assign sampling jobs to workers. Garage now has
three types of `Sampler`s:

- [`LocalSampler`](https://github.com/rlworkgroup/garage/blob/a767dcb95988abc14165a2a320ee39932a1c85ca/src/garage/sampler/local_sampler.py),
the default sampler, which runs workers in the main process. With this sampler,
all the sampling tasks will be run in the same thread.

- [`MultiprocessingSampler`](https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/multiprocessing_sampler.py),
the sampler use [multiprocessing](https://docs.python.org/3/library/multiprocessing.html)
package to run workers in different processes. With multiprocessing, the
sampler can avoid waiting as much as possible.

- [`RaySampler`](https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/ray_sampler.py),
the sampler use [ray](https://github.com/ray-project/ray) framework to run
distributed workers. `RaySampler` can not only run workers in different CPUs,
but also in different machines.

## Worker

Worker is the base unit to perform a rollout and update the agent and the
environment. In paralleling samplers, a worker will typically run in one
exclusive CPU. Garage currently have two kinds of workers:

- [`DefaultWorker`](https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/default_worker.py),
the default worker. It works with a single agent/policy and single environment
in one step.

- [`VecWorker`](https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/vec_worker.py),
the worker with Vectorization. It works with a single agent/policy and multiple
environments. `VecWorker` can compute a batch of actions from a policy
regarding multiple environments to reduce of overhead of sampling.

## Setup Sampler and Worker for a LocalRunner

Setup the sampler and worker for a `LocalRunner` is easy. Just passing
`sampler_cls` and `worker_class` to `runner.setup()` when setup `LocalRunner`.
You can also specify the number of workers by the parameter `n_workers`. For
example:

```py
from garage.sampler import RaySampler, VecWorker

...
runner.setup(
    algo=algo,
    env=env,
    sampler_cls=RaySampler,
    n_workers=4,
    worker_class=VecWorker
)
...
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*
