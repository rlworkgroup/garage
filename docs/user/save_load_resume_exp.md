# Save, Load and Resume Experiments

This document provides solutions to a variety of use cases
regarding saving, loading and resuming of Garage experiments.

**Contents**:

- [Trainer & TFTrainer for Garage experiment](#Trainer-TFTrainer-for-garage-experiment)
- [Saving & Training Models in an experiment](#saving-training-models-in-an-experiment)
- [Loading Models & Resuming an experiment](#loading-models-resuming-an-experiment)

```python
import garage
from garage.experiment import Trainer, TFTrainer
from garage.experiment.deterministic import set_seed
```

## Trainer & TFTrainer for Garage experiment

`Trainer` class in Garage provides users with a range
of utilities to set up environment and train an algorithm
for an experiment.

```Python
trainer = Trainer()
```

`TFTrainer` inherits `Trainer` class and provides
a default TensorFlow session using Python context.

```Python
with TFTrainer(snapshot_config=ctxt) as trainer:
    ...
```

To perform the save, load and resume operations of
an experiment, `Local_trainer` provides four core functions:

- `Trainer.save`: To save snapshot of a specific epoch.
    This function uses [cloudpickle](https://github.com/cloudpipe/cloudpickle)
    utility for serialization. All kind of Model, tensors and dictionaries
    objects related to the experiment setup and training statistics
    within the epoch with saved using this function.
- `Trainer.setup`: To setup `Trainer` instance for
    algorithm and environment in an experiment.
- `Trainer.train`: To train an algorithm given the training
    parameters.`
- `Trainer.restore`: To restore an experiment from snapsnot.
    This function uses cloudpickle's unplicking utilities to deserialize
    pickled object files and re-setup the environment and retrieve model
    data of a specified epoch.
- `Trainer.resume`: To train an algorithm from a restored
    experiment. This function provides the same interface as train().

## Saving & Training Models in an experiment

In general, saving and training models in an experiment includes
 the following steps:

- Initialize the `Trainer`/ `TFTrainer` instance
- Define the environment and algorithms for an experiment
- Setup the trainer for algorithm and environment with `Trainer.setup`.
- Run the training step with `Trainer.train`

```Python
trainer = Trainer()
env = Env(...)
policy = Policy(...)
algo = Algo(
        env=env,
        policy=policy,
        ...)
trainer.setup(algo, env)
trainer.train(n_epochs=100, batch_size=4000)
```

## Loading Models & Resuming an experiment

In general, loading models and resuming an experiment includes
 the following steps:

- Initialize the `Trainer`/ `TFTrainer` instance
- Restore the algorithm and experiment setup with `Trainer.restore`
- Define for the parameters we want to update during training
- Run the training step with `Trainer.resume`

```Python
# to resume immediately.
trainer = Trainer()
trainer.restore(resume_from_dir)
trainer.resume()

# to resume with modified training arguments.
trainer = Trainer()
trainer.restore(resume_from_dir)
trainer.resume(n_epochs=20)
```

### Example on loading TRPO model & finetuning

To provide a setp-by-step example, we will walk you through how to load
a pre-trained network and resume an experiment with
`TRPO`.

First, we specify the directory of snapshot object. By default,
    snapshot objects stored in under `data/local/experiment/`.

```python
snapshot_dir = 'mysnapshot/'  # specify the path
```

Next, we can load the pre-trained model with `trainer.restore()`
by passing the directory path as an argument.

For fine-tunning, we can update the parameters i.e.
`n_epochs`, `batch_size` for training.

Last but not least, we start the training by
`trainer.resume()` with defined parameters as arguments.

```python

@wrap_experiment
def pre_trained_trpo_cartpole(ctxt=None,
    snapshot_dir='data/local/experiment/trpo_gym_tf_cartpole',
    seed=1):
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        trainer.restore(snapshot_dir)
        trainer.resume(n_epochs=30, batch_size=8000)

```

Let's start training now!

```python
pre_trained_trpo_cartpole(snapshot_dir=snapshot_dir)
```

Congratulation, you successfully load a pre-trained model and
start a new experiment!

Note that the experiment has been restored from the last epoch
and start training from that epoch 10 in the example.

```bash
2020-06-26 13:42:00 | Setting seed to 1
2020-06-26 13:42:00 | Restore from snapshot saved in /garage/examples/jupyter/data
2020-06-26 13:42:00 | -- Train Args --     -- Value --
2020-06-26 13:42:00 | n_epochs             10
2020-06-26 13:42:00 | last_epoch           9
2020-06-26 13:42:00 | batch_size           10000
2020-06-26 13:42:00 | store_episodes       0
2020-06-26 13:42:00 | pause_for_plot       0
2020-06-26 13:42:00 | -- Stats --          -- Value --
2020-06-26 13:42:00 | last_itr             10
2020-06-26 13:42:00 | total_env_steps      101047
2020-06-26 13:42:00 | Obtaining samples...
2020-06-26 13:42:00 | epoch #10 | Obtaining samples for iteration 10...
2020-06-26 13:42:02 | epoch #10 | Logging diagnostics...
2020-06-26 13:42:02 | epoch #10 | Optimizing policy...
2020-06-26 13:42:02 | epoch #10 | Computing loss before
2020-06-26 13:42:02 | epoch #10 | Computing KL before
2020-06-26 13:42:02 | epoch #10 | Optimizing
2020-06-26 13:42:02 | epoch #10 | Start CG optimization:
#parameters: 1282, #inputs: 53, #subsample_inputs: 53
2020-06-26 13:42:02 | epoch #10 | computing loss before
2020-06-26 13:42:02 | epoch #10 | computing gradient
2020-06-26 13:42:02 | epoch #10 | gradient computed
2020-06-26 13:42:02 | epoch #10 | computing descent direction
2020-06-26 13:42:03 | epoch #10 | descent direction computed
2020-06-26 13:42:03 | epoch #10 | backtrack iters: 0
2020-06-26 13:42:03 | epoch #10 | optimization finished
2020-06-26 13:42:03 | epoch #10 | Computing KL after
2020-06-26 13:42:03 | epoch #10 | Computing loss after
2020-06-26 13:42:03 | epoch #10 | Fitting baseline...
2020-06-26 13:42:03 | epoch #10 | Saving snapshot...
2020-06-26 13:42:03 | epoch #10 | Saved
2020-06-26 13:42:03 | epoch #10 | Time 2.56 s
2020-06-26 13:42:03 | epoch #10 | EpochTime 2.56 s
---------------------------------------  ---------------
EnvExecTime                                   0.263958
Evaluation/AverageDiscountedReturn           85.0247
Evaluation/AverageReturn                    192.377
Evaluation/TerminationRate                    0.113208
Evaluation/Iteration                         10
Evaluation/MaxReturn                        200
Evaluation/MinReturn                         96
Evaluation/NumEpisodes                       53
Evaluation/StdReturn                         23.1755
Extras/EpisodeRewardMean                    195.81
LinearFeatureBaseline/ExplainedVariance       0.898543
PolicyExecTime                                1.18658
ProcessExecTime                               0.069674
TotalEnvSteps                            111243
policy/Entropy                                0.600691
policy/KL                                     0.00794985
policy/KLBefore                               0
policy/LossAfter                              0.00548463
policy/LossBefore                             0.0165238
policy/Perplexity                             1.82338
policy/dLoss                                  0.0110392
....

---------------------------------------  ---------------
```

You may find the full example at `examples/tf/trpo_gym_tf_cartpole_pretrained.py`.

----
*This page was authored by Iris Liu ([@irisliucy](https://github.com/irisliucy)).*
