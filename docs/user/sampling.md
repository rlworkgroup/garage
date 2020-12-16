# Sampling

`Trainer` gets episodes through Sampling to train the policy. In
Garage, [`Trainer`](https://github.com/rlworkgroup/garage/blob/a767dcb95988abc14165a2a320ee39932a1c85ca/src/garage/trainer.py)
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
With this sampler, all the sampling tasks will run in the same thread.

- [`RaySampler`](https://github.com/rlworkgroup/garage/blob/master/src/garage/sampler/ray_sampler.py),
the sampler using [Ray](https://github.com/ray-project/ray) framework to run
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

## Construct a Sampler

A sampler can be constructed either from parameters directly, or from a worker
factory (the class that can construct workers).

### From parameters

To construct a sampler, we need the following parameters:

- `agents`, the agents/policy we use to make samples.

- `envs`, the environments interacting with policies.

- Parameters for the worker:

  - `max_episode_length`, the maximum length of episodes which will be
sampled.

  - Other optional parameters defining the properties of workers, such as
whether a TensorFlow session is needed, the type of workers, the number of
workers, etc.

```py
from garage.sampler import RaySampler, VecWorker
from garage.tf.policies import GaussianMLPPolicy

env = ...
policy = GaussianMLPPolicy(...)
sampler = RaySampler(agents=policy,
                     envs=env,
                     # params below are for the worker
                     max_episode_length=100,
                     is_tf_worker=True,
                     n_workers=4.
                     worker_class=VecWorker,
                     worker_args=dict(n_envs=12))
```

In this example, we construct a `RaySampler` directly, and the sampler will
construct 4 `VecWorker` when sampling. The maximum length of the episodes
collected will be 100. Besides, we set the level of vectorization (i.e. the
number of environments simulated in one step) to 12 by setting `n_envs` in
`worker_args` for the `VecWorker`.

Noted that as the policy (tf/GaussianMLPPolicy) we use need a TensorFlow
session, we need to set `is_tf_worker` to be `True`.

### From a Worker Factory

```py
from garage.sampler import LocalSampler, WorkerFactory
from garage.torch.policies import GaussianMLPPolicy

env = ...
policy = GaussianMLPPolicy(...)
worker_factory = WorkerFactory(max_episode_length=100,
                               n_workers=4)
sampler = LocalSampler.from_worker_factory(worker_factory=worker_factory,
                                           agents=policy,
                                           envs=env)
```

In the above example, we firstly construct a worker factory, which will
construct 4 workers in the sampler. And the max length of episodes collected
by these workers will be 100. Here we don't choose a type of worker explicitly,
so it will construct `DefaultWorker` by default.

With the worker factory, then we construct a `LocalSampler` with policies and
environments that will be used in sampling.

## Setup Sampler for a Trainer

When we construct the sampler in the launcher. We just need to pass it to the
algorithm object. The algorithm will have a field named `sampler` to save the
sampler. Then the trainer will be able to use the sampler for sampling.

```py

env = ...
sampler = ...
algo = TRPO(...
            sampler=sampler,
            ...)
trainer.setup(algo=algo, env=env)

```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*
