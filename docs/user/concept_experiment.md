# Experiment

The experiment in Garage is a function we use to run an algorithm. This
function is wrapped by a decorator called `wrap_experiment`, which defines the
scope of an experiment, sets up the log directory and what to be saved in
snapshots.

Below is a simple experiment launcher. The first parameter of the experiment
function must be `ctxt`, which is used to pass the experiment's context into
the inner function.

```py
from garage import wrap_experiment

@wrap_experiment
def my_first_experiment(ctxt=None):
    print('Hello World!')

my_first_experiment()
```

Run the example launcher will generate outputs like the following (`CUR_DIR` is
the current directory).

```sh
2020-08-20 15:18:53 | [my_first_experiment] Logging to CUR_DIR/data/local/experiment/my_first_experiment
Hello World!
```

The followings are some useful parameters of `wrap_experiment`. You can see
details of its parameters [here](../_autoapi/garage/index.html#garage.wrap_experiment).

- log_dir: The custom log directory to log to.
- snapshot_mode: Policy for which snapshots to keep. The last iteration will be
saved by default.
- snapshot_gap: Gap between snapshot iterations. Waits this number of
iterations before taking another snapshot.

Here is an example to set a custom log directory:

```py
from garage import wrap_experiment

@wrap_experiment(log_dir='my_custom_log_fir')
def my_first_experiment(ctxt=None):
    print('Hello World!')
```

You can check [this user guide](experiments) for how to write and run an
experiment in detail.

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*
