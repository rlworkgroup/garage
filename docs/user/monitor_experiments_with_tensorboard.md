# Monitor Your Experiments with TensorBoard

[TensorBoard](https://www.tensorflow.org/tensorboard) is a powerful tool that allows you to monitor and visualize an experiment. This section introduces how to set up TensorBoard logging in garage, and how to view/share your TensorBoard logs.

## Prerequisite
The TensorBoard package should be installed upon garage's installation. It can be also manually installed via:
```bash
$ pip install tensorboard
```

## Logger Introduction and Setup
Logging in garage is done via the [dowel](https://github.com/rlworkgroup/dowel) module.

To start using dowel, import `logger`, dowel's logging facility:

`from dowel import logger`

`logger` takes in many different types of input and directs them to the correct output.

Let's start by adding some outputs to logger:

```bash
from dowel import StdOutput, TextOutput, CsvOutput

logger.add_output(StdOutput())  # Add std output
logger.add_output(TextOutput('log_folder/log.txt'))  # Add text output
logger.add_output(CsvOutput('log_folder/table.csv'))  # Add csv output
```

Then call `logger.log()` to direct input to all outputs that accept its type.

For example: `logger.log('test')` will log to both StdOutput and TextOutput, but not to CsvOutput (since it doesn't handle string output):
```bash
                    +---------+
       +---'test'--->StdOutput|
       |            +---------+
       |
+------+            +----------+
|logger+---'test'--->TextOutput|
+------+            +----------+
       |
       |            +---------+
       +-----!!----->CsvOutput|
                    +---------+
```

You can find more about `logger` and its supported types [here](https://github.com/rlworkgroup/dowel/blob/master/src/dowel/logger.py).

## Log TensorBoard with `logger`
This section introduces how to log TensorBoard output with `logger`, and how to log different types of TensorBoard output.

First, add TensorBoard output type to `logger`. Pass `log_dir` as the save location of the TensorBoard event files :
```bash
logger.add_output(dowel.TensorBoardOutput(log_dir))
```

### Log **Scalar** Values to TensorBoard

You would need [`dowel.TabularInput`](https://github.com/rlworkgroup/dowel/blob/master/src/dowel/tabular_input.py).

To prepare your TensorBoard output, import dowel's TabularInput instance `tabular` by:

`from dowel import tabular`

Add scalar values:
```bash
tabular.record('Epoch', epoch)
tabular.record('# Sample', i_sample)
tabular.record('AverageDiscountedReturn', return)
```

Finally, log your scalar values with `logger`:

`logger.log(tabular)`

### Log **histograms** to Tensorboard

To log histograms to TensorBoard, you would need to pass [`dowel.Histogram`](https://github.com/rlworkgroup/dowel/blob/master/src/dowel/histogram.py) to `tabular`.

`dowel.Histogram` is implemented as a typed view of a numpy array. It will accept input that `numpy.asarray` will.

For example:
```bash
from dowel import Histogram, logger, tabular

samples = norm.rvs(100)  # ndarray of 100 samples of a distribution
hist = Histogram(samples)
tabular.record('Sample', hist)

logger.log(tabular)
```

## The "Automatic" Way: Setup Your Experiment with `@wrap_experiment`
Garage provides a decorator `@wrap_experiment` to help the logger setup process.

Suppose you have an experiment function `my_experiment()` with the following definition:
```bash
def my_experiment(seed=1):
```
Here's how you can enable logging for this experiment:

First, import the `wrap_experiment` decorator function.
```bash
from garage import wrap_experiment
```

Now wrap your function:
```bash
@wrap_experiment
def my_experiment(ctxt=None, seed=1):
    """
    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
    """
```
`@wrap_experiment` helps create experiment log directories and set up logger outputs (e.g. `logger.add_output(dowel.TensorBoardOutput(log_dir))`). After the setup process, `my_experiment()` is called to start the actual experiment.

Note that `my_experiment()` now receives a new input `ctxt`. This is required for  the wrapped function to receive experiment configuration created by the wrapper.

This decorater can also be invoked **with arguments**:

```bash
@wrap_experiment(log_dir='my_dir')
def my_experiment(ctxt, seed):
```
For a complete list of arguments, you can refer to the definition of `wrap_experiment` [here](https://github.com/rlworkgroup/garage/blob/master/src/garage/experiment/experiment.py).

Congrats! Now you have finished setting up TensorBoard logging. You can run your experiment and find your TensorBoard logs under the log directory.

By default, your `log_dir` is `garage/data/local/experiment/{experiment_name}_{experiment_id}`

For our example experiment `my_experiment`, the default log directory is `garage/data/local/experiment/my_experiment`.

Now if you run the experiment again, the logs of the second run will be saved under `garage/data/local/experiment/my_experiment_1`.

## View Your TensorBoard Logs

You can view your TensorBoard results via the TensorBoard command line tool:

`$ tensorboard --logdir logs`

Once you run the command, you will see
```bash
$ tensorboard --logdir logs
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.2.2 at http://localhost:6006/ (Press CTRL+C to quit)
```

Open your browser and navigate to http://localhost:6006/ to see a dashboard of your TensorBoard results.

Note that if your `--logdir` directory contains multiple TensorBoard logs, the dashboard will show all of them. This is useful when comparing results from multiple runs.

You can find more TensorBoard dashboard details [here](https://www.tensorflow.org/tensorboard/get_started).

## Share Your TensorBoard Logs

[TensorBoard Dev](https://tensorboard.dev/) allows you to share your TensorBoard results with others.

Upload your logs by:
```bash
$ tensorboard dev upload --logdir logs \
    --name "(optional) My latest experiment" \
    --description "(optional) Simple comparison of several hyperparameters"
```

You should see:
```bash
$ tensorboard dev upload --logdir logs

Upload started and will continue reading any new data as it's added
to the logdir. To stop uploading, press Ctrl-C.
View your TensorBoard live at: https://tensorboard.dev/experiment/MPTLRxtDQVGp9t4DQsleHQ/
```

Now you can go to the link provided to view your experiment, or share it with others.

Find more about TensorBoard Dev [here](https://tensorboard.dev/#get-started).
