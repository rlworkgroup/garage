# Use a pre-trained network to start a new experiment

In general starting a new experiment with a pre-trained network follow the same few steps:

- Initialize the pretrained model
- Reshape the final layer(s) to have the same number of outputs as the number of classes in the new dataset
- Define for the optimization algorithm which parameters we want to update during training
- Run the training step

In this tutorial, we will walk you through how to use a pre-trained network to start a new experiment with `TRPO algorithm` in garage. In practice, garage allows users to save and load a pre-trained model to perform the task of interest in reinforcement learning.

```python
import garage
from garage.experiment import LocalTFRunner
from garage.experiment.deterministic import set_seed
```

## load and train a pre-trained model

### Use garage snapshot

To load the previously trained model, one recommended way is to use garage snapshot, a `Snapshotter` instance that is directly used when we run experiments with `LocalRunner` or `LocalTFRunner`. A snapshot of the model and its data is automatically saved to the local directory (by dafault: `data/local/experiment/`) as a new experiment is run.

In general starting a new experiment with a pre-trained network follow the same few steps:

- Initialize the `Snapshotter` instance
- Restore the pre-trained model
- Define for the parameters we want to update during training
- Run the training step with `resume`

To load a pre-trained model stored as an `Snapshotter` instance, we simply need to pass a defined snapshot configuration as a parameter `snapshot_config` to a `LocalRunner` instance.

Define snapshot configuration as follows:

- snapshot_dir: directory of snapshot object
- snapshot_mode: type of snapshot
- snapshot_gap: gap number of snapshot

```python
snapshot_dir = 'mysnapshot/'  # specify the path
ctxt = garage.experiment.SnapshotConfig(snapshot_dir=snapshot_dir,
                                      snapshot_mode='last', # only last iteration will be saved
                                      snapshot_gap=1)
```

`Snapshotter` object is initialized once `ctxt` is passed to the `LocalTFRunner` instance. If `snapshot_config` is set to`None`, a default snapshot configuration will be used.

Next, we can load the pre-trained model with `runner.restore()` by passing the directory path as an argument.

For fine-tunning, we can update the parameters i.e. `n_epochs`, `batch_size` for training.

Last but not least, we start the training by `runner.resume()` with defined parameters as arguments.

```python

def pre_trained_trpo_cartpole(snapshot_dir):
    with LocalTFRunner(snapshot_config=ctxt) as runner:
        runner.restore(snapshot_dir)
        runner.resume(n_epochs=20)

```

Let's start training now!

```python
pre_trained_trpo_cartpole(snapshot_dir)
```

Congratulation, you successfully load a pre-trained model and start a new experiment!

```bash
2020-06-26 13:42:00 | Setting seed to 1
2020-06-26 13:42:00 | Restore from snapshot saved in /garage/examples/jupyter/data
2020-06-26 13:42:00 | -- Train Args --     -- Value --
2020-06-26 13:42:00 | n_epochs             10
2020-06-26 13:42:00 | last_epoch           9
2020-06-26 13:42:00 | batch_size           10000
2020-06-26 13:42:00 | store_paths          0
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
2020-06-26 13:42:02 | epoch #10 | Start CG optimization: #parameters: 1282, #inputs: 53, #subsample_inputs: 53
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
Evaluation/CompletionRate                     0.113208
Evaluation/Iteration                         10
Evaluation/MaxReturn                        200
Evaluation/MinReturn                         96
Evaluation/NumTrajs                          53
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
---------------------------------------  ----------------
2020-06-26 13:42:16 | epoch #18 | Obtaining samples for iteration 18...
2020-06-26 13:42:17 | epoch #18 | Logging diagnostics...
2020-06-26 13:42:17 | epoch #18 | Optimizing policy...
2020-06-26 13:42:17 | epoch #18 | Computing loss before
2020-06-26 13:42:17 | epoch #18 | Computing KL before
2020-06-26 13:42:17 | epoch #18 | Optimizing
2020-06-26 13:42:17 | epoch #18 | Start CG optimization: #parameters: 1282, #inputs: 52, #subsample_inputs: 52
2020-06-26 13:42:17 | epoch #18 | computing loss before
2020-06-26 13:42:17 | epoch #18 | computing gradient
2020-06-26 13:42:17 | epoch #18 | gradient computed
2020-06-26 13:42:17 | epoch #18 | computing descent direction
2020-06-26 13:42:18 | epoch #18 | descent direction computed
2020-06-26 13:42:18 | epoch #18 | backtrack iters: 0
2020-06-26 13:42:18 | epoch #18 | optimization finished
2020-06-26 13:42:18 | epoch #18 | Computing KL after
2020-06-26 13:42:18 | epoch #18 | Computing loss after
2020-06-26 13:42:18 | epoch #18 | Fitting baseline...
2020-06-26 13:42:18 | epoch #18 | Saving snapshot...
2020-06-26 13:42:18 | epoch #18 | Saved
2020-06-26 13:42:18 | epoch #18 | Time 17.29 s
2020-06-26 13:42:18 | epoch #18 | EpochTime 1.86 s
---------------------------------------  ---------------
EnvExecTime                                   0.266177
Evaluation/AverageDiscountedReturn           86.602
Evaluation/AverageReturn                    200
Evaluation/CompletionRate                     0
Evaluation/Iteration                         18
Evaluation/MaxReturn                        200
Evaluation/MinReturn                        200
Evaluation/NumTrajs                          52
Evaluation/StdReturn                          0
Extras/EpisodeRewardMean                    199.98
LinearFeatureBaseline/ExplainedVariance       0.999972
PolicyExecTime                                1.18185
ProcessExecTime                               0.0698261
TotalEnvSteps                            193658
policy/Entropy                                0.508267
policy/KL                                     0.00908262
policy/KLBefore                               0
policy/LossAfter                             -0.00815415
policy/LossBefore                            -1.2031e-07
policy/Perplexity                             1.66241
policy/dLoss                                  0.00815403
---------------------------------------  ---------------
2020-06-26 13:42:18 | epoch #19 | Obtaining samples for iteration 19...
2020-06-26 13:42:19 | epoch #19 | Logging diagnostics...
2020-06-26 13:42:19 | epoch #19 | Optimizing policy...
2020-06-26 13:42:19 | epoch #19 | Computing loss before
2020-06-26 13:42:19 | epoch #19 | Computing KL before
2020-06-26 13:42:19 | epoch #19 | Optimizing
2020-06-26 13:42:19 | epoch #19 | Start CG optimization: #parameters: 1282, #inputs: 52, #subsample_inputs: 52
2020-06-26 13:42:19 | epoch #19 | computing loss before
2020-06-26 13:42:19 | epoch #19 | computing gradient
2020-06-26 13:42:19 | epoch #19 | gradient computed
2020-06-26 13:42:19 | epoch #19 | computing descent direction
2020-06-26 13:42:20 | epoch #19 | descent direction computed
2020-06-26 13:42:20 | epoch #19 | backtrack iters: 4
2020-06-26 13:42:20 | epoch #19 | optimization finished
2020-06-26 13:42:20 | epoch #19 | Computing KL after
2020-06-26 13:42:20 | epoch #19 | Computing loss after
2020-06-26 13:42:20 | epoch #19 | Fitting baseline...
2020-06-26 13:42:20 | epoch #19 | Saving snapshot...
2020-06-26 13:42:20 | epoch #19 | Saved
2020-06-26 13:42:20 | epoch #19 | Time 19.25 s
2020-06-26 13:42:20 | epoch #19 | EpochTime 1.96 s
---------------------------------------  ---------------
EnvExecTime                                   0.286301
Evaluation/AverageDiscountedReturn           86.602
Evaluation/AverageReturn                    200
Evaluation/CompletionRate                     0
Evaluation/Iteration                         19
Evaluation/MaxReturn                        200
Evaluation/MinReturn                        200
Evaluation/NumTrajs                          52
Evaluation/StdReturn                          0
Extras/EpisodeRewardMean                    200
LinearFeatureBaseline/ExplainedVariance       0.999973
PolicyExecTime                                1.24483
ProcessExecTime                               0.076405
TotalEnvSteps                            204058
policy/Entropy                                0.529518
policy/KL                                     0.00595805
policy/KLBefore                               0
policy/LossAfter                             -0.00285657
policy/LossBefore                            -2.5089e-07
policy/Perplexity                             1.69811
policy/dLoss                                  0.00285632
---------------------------------------  ---------------
```

You may find the full example under `examples/jupyter/pre-trained_trpo_gym_tf_cartpole.ipynb`.
