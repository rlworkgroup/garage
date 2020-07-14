# Sampling

`LocalRunner` gets trajectories through Sampling to train the policy. In
Garage, [`LocalRunner`](https://github.com/rlworkgroup/garage/blob/a767dcb95988abc14165a2a320ee39932a1c85ca/src/garage/experiment/local_runner.py)
uses [`Sampler`](https://github.com/rlworkgroup/garage/blob/a767dcb95988abc14165a2a320ee39932a1c85ca/src/garage/sampler/sampler.py)
to perform sampling. `Sampler` manages [`Worker`](https://github.com/rlworkgroup/garage/blob/a767dcb95988abc14165a2a320ee39932a1c85ca/src/garage/sampler/worker.py)s
and assign specific tasks to them, which is doing rollouts with agents and
environments. You can also implement your own sampler and worker. The
followings introduce the existing samplers and workers in Garage.

## Sampler

Sampler is responsible for assign sampling jobs to workers. Garage now has
two types of `Sampler`s:

- [`LocalSampler`](https://github.com/rlworkgroup/garage/blob/a767dcb95988abc14165a2a320ee39932a1c85ca/src/garage/sampler/local_sampler.py),
the default sampler, which runs workers in the main process in serial style.
With this sampler, all the sampling tasks will be run in the same thread.

- [`RaySampler`](https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/ray_sampler.py),
the sampler using [ray](https://github.com/ray-project/ray) framework to run
distributed workers in parallel style. `RaySampler` can not only run workers in
different CPUs, but also in different machines across network.

## Worker

Worker is the basic unit to perform a rollout per step. In paralleling
samplers, each worker will typically run in one exclusive CPU. For most
algorithms, Garage provides two kinds of workers, `DefaultWorker` and
`VecWorker`. A few algorithms (RL2 and PEARL) use custom workers specific to
that algorithm.

- [`DefaultWorker`](https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/default_worker.py),
the default worker. It works with one single agent/policy and one single
environment in one step.

- [`VecWorker`](https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/vec_worker.py),
the worker with Vectorization, which runs multiple instances of the simulation
on a single CPU. `VecWorker` can compute a batch of actions from a policy
regarding multiple environments to reduce of overhead of sampling (e.g. feeding
forward a neural network).

## Setup Sampler and Worker for a LocalRunner

Setup the sampler and worker for a `LocalRunner` is easy. Just passing
`sampler_cls` and `worker_class` to `runner.setup()` when setup `LocalRunner`.
The number of workers in the sampler can be set by the parameter `n_workers`.

For `VecWorker`, you can set the level of vectorization (i.e. the number of
environments simulated in one step) by setting `n_envs` in `worker_args`.

```py
from garage.sampler import RaySampler, VecWorker

...
runner.setup(
    algo=algo,
    env=env,
    sampler_cls=RaySampler,
    n_workers=4,
    worker_class=VecWorker,
    worker_args=dict(n_envs=12)
)
...
```

In the above example, we choose `RaySampler` and `VecWorker`, set the number
of workers to 4, and set the level of vectorization to 12. With this
configuration, sampling will run in 4 CPUs in parallel and each worker will
sample 12 actions in one step.

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*
