# Logging and plotting

## Logging

garage supports convenient and useful logging. garage uses [dowel](https://github.com/rlworkgroup/dowel)
for logging. The `logger` supports many outputs, including

- Std output
- Text output
- Csv output
- TensorBoard output

In garage's experiment, the `logger` will output to all these four.

Here is an example of logging in garage.

```py
from garage import wrap_experiment
from dowel import logger, tabular

@wrap_experiment
def log_experiment(ctxt=None):
    for i in range(100):
        # Log str directly
        logger.log('Logging messages:')
        # Log scalar values with the key 'AverageReturn'
        tabular.record('AverageReturn', i)

        # The Trainer will do these steps for you, if you log things in
        # the algorithms.
        logger.log(tabular)
        logger.dump_all()

log_experiment()
```

Running the example will generate outputs like:

```sh
2020-10-21 14:06:04 | [log_experiment] Logging to [CUR_DIR]/data/local/experiment/log_experiment
2020-10-21 14:06:04 | [log_experiment] Logging messages:
-------------  -
AverageReturn  0
-------------  -
2020-10-21 14:06:04 | [log_experiment] Logging messages:
-------------  -
AverageReturn  1
-------------  -
2020-10-21 14:06:04 | [log_experiment] Logging messages:
-------------  -
AverageReturn  2
-------------  -
```

To look output with TensorBoard, you can refer this [page](monitor_experiments_with_tensorboard).

To set a customized log directory, just pass a `log_dir` argument to the
experiment.

```py
@wrap_experiment(log_dir='my_custom_log_fir')
```

## Plotting

In garage, as long as the environment implement the `visualize()` method, is
it easy to plot a policy running in the environment when training.

To visualize an experiment, just set the `plot` argument to `True` in the
[`train`](../_autoapi/garage/index.html#garage.Trainer.train) method of
`Trainer`. For example, in [example/tf/trpo_cartpole.py](https://github.com/rlworkgroup/garage/blob/master/examples/tf/trpo_cartpole.py),
change the train line into:

```py
trainer.train(n_epochs=100, batch_size=4000, plot=True)
```

If you want to pause in every epoch, just set `pause_for_plot` to `True`.

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*
