# Monitor Your Experiments with TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a powerful tool that
allows you to monitor and visualize an experiment.
This section introduces how to set up TensorBoard logging in garage, and how to
view/share your TensorBoard logs.

## Prerequisite

The TensorBoard package should be installed upon garage's installation.
It can be also manually installed via:

```bash
pip install tensorboard
```

### Setup Your Experiment

Use `@wrap_experiment` to setup your experiment.
This will automatically setup TensorBoard output and allocate an experiment
directory for you.

Find more about how to use `@wrap_experiment` [here](experiments).

## Log to TensorBoard

This section introduces how to log TensorBoard output with `logger`, and how to
log different types of TensorBoard output.

Find more about how to use `logger` [here](https://github.com/rlworkgroup/dowel/blob/master/src/dowel/logger.py).

### Log **Scalar** Values to TensorBoard

You would need [`dowel.TabularInput`](https://github.com/rlworkgroup/dowel/blob/master/src/dowel/tabular_input.py).

To prepare your TensorBoard output, import dowel's `TabularInput` instance
`tabular` by:

```py
from dowel import tabular
```

Add scalar values:

```py
tabular.record('Epoch', epoch)
tabular.record('# Sample', i_sample)
tabular.record('AverageDiscountedReturn', return)
```

Finally, log your scalar values with `logger`:

```py
logger.log(tabular)
```

### Log **Histograms** to TensorBoard

To log histograms to TensorBoard, you would need to pass
[`dowel.Histogram`](https://github.com/rlworkgroup/dowel/blob/master/src/dowel/histogram.py)
to `tabular`.

`dowel.Histogram` is implemented as a typed view of a numpy array.
It will accept input that `numpy.asarray` will.

For example:

```py
from dowel import Histogram, logger, tabular

samples = norm.rvs(100)  # ndarray of 100 samples of a normal distribution
hist = Histogram(samples)
tabular.record('Sample', hist)

logger.log(tabular)
```

You can use `Histogram()` to log distribution types from
[`scipy.stats`](https://docs.scipy.org/doc/scipy/reference/stats.html).

## View Your TensorBoard Logs

You can view your TensorBoard results via the TensorBoard command line tool:

`$ tensorboard --logdir logs`

Once you run the command, you will see

```bash
$ tensorboard --logdir logs
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.2.2 at http://localhost:6006/ (Press CTRL+C to quit)
```

Open your browser and navigate to the link to see a dashboard of your
TensorBoard results.

Note that if your `--logdir` directory contains multiple TensorBoard logs,
the dashboard will show all of them. This is useful when comparing results from
multiple runs.

You can find more TensorBoard dashboard details
[here](https://www.tensorflow.org/tensorboard/get_started).

## Share Your TensorBoard Logs

[TensorBoard Dev](https://tensorboard.dev/) allows you to share your TensorBoard
 results with others.

Upload your logs by:

```bash
$ tensorboard dev upload --logdir data/log \
    --name "(optional) My latest experiment" \
    --description "(optional) Simple comparison of several hyperparameters"
```

You should see:

```bash
$ tensorboard dev upload --logdir data/log

Upload started and will continue reading any new data as it's added
to the logdir. To stop uploading, press Ctrl-C.
View your TensorBoard live at: https://tensorboard.dev/experiment/MPTLRxtDQVGp9t4DQsleHQ/
```

Now you can go to the link provided to view your experiment, or share it with
others.

Find more about TensorBoard Dev [here](https://tensorboard.dev/#get-started).

## Advanced Features

### Change the output directory or x-axis

By default, `@wrap_experiment` sets up TensorBoard output via:

```py
logger.add_output(dowel.TensorBoardOutput(log_dir, x_axis='TotalEnvSteps'))
```

The default `log_dir` is `$(pwd)/data/local`, and the `x_axis` is
`'TotalEnvSteps'`.

To manually output to a different directory, you can
set up `logger` in your experiment by:

```py
logger.add_output(dowel.TensorBoardOutput(MyLogDir))
```

### Add additional x-axes

If you'd like to view your logs on more than one x-axis, you can configure dowel
to log to additional x-axes by passing a list of tabular keys to the
`additional_x_axes` parameter of `TensorBoardOutput`.

```py
logger.add_output(dowel.TensorBoardOutput(
    log_dir,
    x_axis='TotalEnvSteps',
    additional_x_axes=['Itr'],  # Logs keys by optimization iteration as well
))
```

Using this configuration, the key `Loss` will appear in TensorBoard under both
the tag `Loss` using `TotalEnvSteps` as its x-axis, and under the tag `Loss/Itr`
using `Itr` as its x-axis.

----

*This page was authored by Eric Yihan Chen
([@AiRuiChen](https://github.com/AiRuiChen)).*
