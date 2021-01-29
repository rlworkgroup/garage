# Quick Start with garage

Table of Content
- [What is garage?](#what-is-garage)
- [Why garage?](#why-garage)
- [Kick Start garage](#kick-start-garage)
- [Open Source Support](#open-source-support)
- [Resources](#resources)


## What is garage?

garage is a reinforcement learning (RL) toolkit for developing and evaluating algorithms. The garage library also provides a collection of state-of-the-art implementations of RL algorithms.

The toolkit provides a wide range of modular tools for implementing RL algorithms, including:

- Composable neural network models
- Replay buffers
- High-performance samplers
- An expressive experiment definition interface
- Tools for reproducibility (e.g. set a global random seed which all components respect)
- Logging to many outputs, including TensorBoard
- Reliable experiment checkpointing and resuming
- Environment interfaces for many popular benchmark suites
- Supporting for running garage in diverse environments, including always up-to-date Docker containers

## Why garage?

garage aims to provide both researchers and developers:

- a flexible and structured tool for developing algorithms to solve a variety of RL problems,
- a standardized and [reproducible](ensure_your_experiments_are_reproducible.md) environment for experimenting and evaluating RL algorithms,
- a collection of benchmarks and examples of RL algorithms.

## Kick Start garage

This quickstart will show how to quickly get started with garage in 5 minutes.

```py
import garage
```

### Algorithms

An array of algorithms are available in garage:

```eval_rst
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|        Algorithm       |                                                                         Framework(s)                                                                         |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|           CEM          |                                             `Numpy <../_autoapi/garage/np/algos/index.html#garage.np.algos.CEM>`_                                            |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|         CMA-ES         |                                            `Numpy <../_autoapi/garage/np/algos/index.html#garage.np.algos.CMAES>`_                                           |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
| REINFORCE (a.k.a. VPG) |   `PyTorch <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.VPG>`_, `TensorFlow <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.VPG>`_  |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          DDPG          | `PyTorch <..//_autoapi/garage/torch/algos/index.html#garage.torch.algos.DDPG>`_, `TensorFlow <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.DDPG>`_ |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|           DQN          |                                          `TensorFlow <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.DQN>`_                                          |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          DDQN          |                                                                          TensorFlow                                                                          |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          ERWR          |                                          `TensorFlow <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.ERWR>`_                                         |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|           NPO          |                                          `TensorFlow <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.NPO>`_                                          |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|           PPO          |                                   `PyTorch <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.PPO>`_, TensorFlow                                  |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          REPS          |                                          `TensorFlow <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.REPS>`_                                         |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|           TD3          |                                          `TensorFlow <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.TD3>`_                                          |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          TNPG          |                                          `TensorFlow <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.TNPG>`_                                         |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          TRPO          |                                  `PyTorch <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.TRPO>`_, TensorFlow                                  |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          MAML          |                                                                            PyTorch                                                                           |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|           RL2          |                                          `TensorFlow <../_autoapi/garage/tf/algos/index.html#garage.tf.algos.RL2>`_                                          |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          PEARL         |                                        `PyTorch <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.PEARL>`_                                       |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|           SAC          |                                         `PyTorch <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.SAC>`_                                        |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          MTSAC         |                                                                            PyTorch                                                                           |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|          MTPPO         |                                                                      PyTorch, TensorFlow                                                                     |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|         MTTRPO         |                                                                      PyTorch, TensorFlow                                                                     |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|     Task Embedding     |                                                                          TensorFlow                                                                          |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
|   Behavioral Cloning   |                                         `PyTorch <../_autoapi/garage/torch/algos/index.html#garage.torch.algos.BC>`_                                         |
+------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------+
```

They are organized in the [github repository](https://github.com/rlworkgroup/garage/tree/master/src/garage) as:

```bash
└── garage
    ├── envs
    ├── experiment
    ├── misc
    ├── np
    ├── plotter
    ├── replay_buffer
    ├── sampler
    ├── tf
    └── torch
```

Note: clickable links represents the directory of algorithms.

A simple pytorch example to import `TRPO` algorithm, as well as, the policy
`GaussianMLPPolicy`, value function `GaussianMLPValueFunction` and sampler
`LocalSampler` in garage is shown below:

```py
import gym
import torch

from garage.envs import GarageEnv, normalize
from garage.sampler import LocalSampler
from garage.torch.algos import TRPO as PyTorch_TRPO
from garage.torch.policies import GaussianMLPPolicy as PyTorch_GMP
from garage.torch.value_functions import GaussianMLPValueFunction

def trpo_garage_pytorch():

    env = GarageEnv(normalize(gym.make(env_id))) # specify env_id

    policy = PyTorch_GMP(env.spec,
                         hidden_sizes= [32, 32],
                         hidden_nonlinearity=torch.tanh,
                         output_nonlinearity=None)

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    sampler = LocalSampler(agents=policy,
                           envs=env,
                           max_episode_length=env.spec.max_episode_length)

    algo = PyTorch_TRPO(
        env_spec=env.spec,
        policy=policy,
        value_function=value_function,
        sampler=sampler,
        discount=0.99,
        gae_lambda=0.97)

```

The full code can be found [here](https://github.com/rlworkgroup/garage/blob/master/benchmarks/src/garage_benchmarks/experiments/algos/trpo_garage_pytorch.py).

To know more about implementing new algorithms, see [this guide](https://garage.readthedocs.io/en/latest/user/implement_algo.html)

### Running Examples

Garage ships with example files to help you get started. To get a
list of examples, run:

```
garage examples
```

This prints a list of examples along with their fully qualified name, such as:

```
tf/dqn_cartpole.py (garage.examples.tf.dqn_cartpole.py)
```

To get the source of an example, run:

```
garage examples tf/dqn_cartpole.py
```

This will print the source on your console, which you can write to a file as
follows:

```
garage examples tf/dqn_cartpole.py > tf_dqn_cartpole.py
```

You can also directly run an example by passing the fully qualified name to
`python -m`, as follows:

```
python -m garage.examples.tf.dqn_cartpole.py
```

You can also access the examples for a specific version on GitHub by visiting
the tag corresponding to that version and then navigating to
`src/garage/examples`.

### Running Experiments

In garage, experiments are run using the "experiment launcher" `wrap_experiment`, a decorated Python function, which can be imported directly from the garage package.

```py
from garage import wrap_experiment
```

Moreover, objects, such as `trainer`, `environment`, `policy`, `sampler` e.t.c are commonly used when constructing experiments in garage.

```eval_rst
.. literalinclude:: ../../benchmarks/src/garage_benchmarks/experiments/algos/trpo_garage_pytorch.py
```

[This page](https://garage.readthedocs.io/en/latest/user/experiments.html) will give you more insight into running experiments.

### Plotting results

In garage, we use [TensorBoard](https://www.tensorflow.org/tensorboard) for plotting experiment results.

[This guide](https://garage.readthedocs.io/en/latest/user/monitor_experiments_with_tensorboard.html) will provide details how to set up tensorboard when running experiments in garage.

### Experiment outputs

`Localrunner` is a state manager of experiments in garage, It is set up to create, save and restore the state, also known as `snapshot` object, upon/ during an experiment. The `snapshot` object includes the hyperparameter configuration, training progress, a pickled object of algorithm(s) and environment(s), tensorboard event file etc.

Experiment results will, by default, output to the same directory as the garage package in the relative directory `data/local/experiment`. The output directory is generally organized as the following:

```bash
└── data
    └── local
        └── experiment
            └── your_experiment_name
                ├── progress.csv
                ├── debug.log
                ├── variant.json
                ├── metadata.json
                ├── launch_archive.tar.xz
                └── events.out.tfevents.xxx
```

`wrap_experiment` can be invoked with arguments to support actions like modifying default output directory, changing snapshot modes, controlling snapshot gap etc. For example, to modify the default output directory and change the snapshot mode from `last` (only last iteration will be saved) to `all`, we can do this:

```py
@wrap_experiment(log_dir='./your_log_dir', snapshot_mode='all')
        def my_experiment(ctxt, seed, lr=0.5):
            ...
```

During an experiment, garage extensively use `logger` from [`Dowel`](https://github.com/rlworkgroup/dowel) for logging outputs to StdOutput, and/ or TextOutput, and/or CsvOutput. For details, you can check [this](https://github.com/rlworkgroup/dowel/blob/master/src/dowel/logger.py).

## Open Source Support

Since October 2018, garage is active in the open-source community contributing to RL researches and developments. Any [contributions](https://garage.readthedocs.io/en/latest/user/preparing_a_pr.html) from the community is more than welcomed.

## Resources

If you are interested in a more in-depth and specific capabilities of garage, you can find many other guides in this website such as, but not limited to, the followings:

- [Installation](https://garage.readthedocs.io/en/latest/user/installation.html)
- [Setting up development environment](https://garage.readthedocs.io/en/latest/user/setting_up_your_development_environment.html)

----

*This page was authored by Iris Liu ([@irisliucy](https://github.com/irisliucy)).*
