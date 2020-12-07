
# Ensure Your Experiments are Reproducible

Ensure the reproducibility of your experiments with comprehensive launcher
files. Launcher files are used to initialize the algorithm, its components
(policy, replay buffer, etc.), and tools for managing experiments.

There are some important details in this example launcher file.

```python
from garage import wrap_experiment
from garage.experiment import deterministic
from garage.trainer import Trainer
import a.b.c

@wrap_experiment(wrap_experiment_args, wrap_experiment_kwargs)
def experiment(ctxt=None, seed):
    deterministic.set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)
    env = YOUR_ENV

    # Initialize algorithm dependencies (policy, Q-functions,
    # replay buffers, etc.)
    policy = ...
    sampler = ...
    algo = ...

    trainer.setup(algo=algo, env=env)
    trainer.train(n_epochs=num_epochs, batch_size=)

seed = 0
experiment(seed)
```

## Important Take Aways

### Use `garage.wrap_experiment` for snapshotting and saving results

Use the function decorator `wrap_experiment`  to enable the storage of results
of your experiment. By default, experiments are named based on the name of the
function that is wrapped by the wrap experiment decorator. In the case of the
example above, experiments would be saved under the directory name
`experiment/`. You can specify the directory in which all results are saved to
using the parameter `log_dir`. You can further modify the directory tree under
which your experiments are saved and the experiment naming notation by using
the parameters `prefix`, `name` , `use_existing_dir` and `name_parameters`.
For example, by setting `name_parameters` to true, the parameters to the
function wrapped by `wrap_experiment` will be incorporated into the directory
name. Using the above example launcher file, experiments would be saved under
the directory name `experiment_seed_*SEED_NUMBER*`. There are defaults to these
parameters that allow for easy to understand organization for most use cases.
The logs for your experiments (csv's, textual outputs, tensorboard logs) will
be saved in this file structure. Lastly, you can save the repo that you
launched your experiment from by enabling `archive_launch_repo`. This is
enabled by default, and your archived repository is saved under the directory
tree.

`wrap_experiment`’s other use is for enabling the snapshotting of your
experiment. Snapshots are `pkl` files that can be used for retrieving the
components of an experiment (policy, algorithm, sampler). They are used to
capture the state of your experiment at any training epoch. You can control
the rate at which you collect snapshots for your experiment using the parameter
`snapshot_mode`. The `snapshot_mode` can be either "all" (all iterations will
be saved), "last" (only the last iteration will be saved), “gap” (every
`snapshot_gap` iterations are saved), or "none" (do not save snapshots).

### Use `deterministic.set_seed`

In machine learning research, setting the seed of the random number generator
that is used by your algorithm and its components will allow you to reproduce
the behavior of your algorithm. If you set the seed to be the same value on
different runs, the behavior of your algorithm will be the same. Vice versa,
setting the seed of your experiment to different numbers will allow you to
observe the different behavior modes of your algorithm and its components
under different sequences of random numbers.
*Note: In experiments using Pytorch+GPU, there is a small runtime performance
cost for using `deterministic.set_seed`*

### Use `garage.trainer.Trainer` for managing your training process

`Trainer` manages the boiler-plate interactions between your sampler,
algorithm, logger, and snapshotter.

To setup the `Trainer` use the function
`Trainer.setup(algo, env, ...)`. It takes your algorithm and environment as
required parameters.
Lastly, to trigger the training process of your experiment, use the function
`Trainer.train(n_epochs, batch_size, ...)`. The required parameter
`n_epochs` is used for specifying the number of training epochs that your
experiment’s algorithm is run for. The parameter `batch_size` is used for
specifying the default number of time steps that will be collected from your
experiment's environment between training epochs.

## Examples

For examples of how to write a launcher file for your experiment, refer to the
launcher files under the `examples` directory.

----

*This page was authored by Avnish Narayan*
*([@avnishn](https://github.com/avnishn))*
