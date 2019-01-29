.. _experiments:


===================
Running Experiments
===================


We use object oriented abstractions for different components required for an experiment. To run an experiment, simply construct the corresponding objects for the environment, algorithm, etc. and call the appropriate train method on the algorithm. A sample script is provided in :code:`examples/trpo_cartpole.py`. The code is also pasted below for a quick glance:

.. code-block:: python

    import gym

    from garage.baselines import LinearFeatureBaseline
    from garage.experiment import run_experiment
    from garage.tf.algos import TRPO
    from garage.tf.envs import TfEnv
    from garage.tf.policies import CategoricalMLPPolicy


    def run_task(*_):
        """Wrap TRPO training task in the run_task function."""
        env = TfEnv(env_name="CartPole-v1")

        policy = CategoricalMLPPolicy(
            name="policy", env_spec=env.spec, hidden_sizes=(32, 32))

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        algo = TRPO(
            env=env,
            policy=policy,
            baseline=baseline,
            batch_size=4000,
            max_path_length=100,
            n_itr=100,
            discount=0.99,
            max_kl_step=0.01,
            plot=False)
        algo.train()


    run_experiment(
        run_task,
        n_parallel=1,
        snapshot_mode="last",
        seed=1,
        plot=False,
    )


You should see some log messages like the following:

.. code-block:: text

    2019-01-31 23:05:34 | Setting seed to 1
    2019-01-31 23:05:34 | tensorboard data will be logged into:/root/code/garage/data/local/experiment/experiment_2019_01_31_23_05_29_0001
    /opt/conda/envs/garage/lib/python3.6/site-packages/gym/envs/registration.py:14: PkgResourcesDeprecationWarning: Parameters to load are deprecated.  Call .resolve and .require separately.
      result = entry_point.load(False)
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Obtaining samples...
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Obtaining samples for iteration 0...
    0% [##############################] 100% | ETA: 00:00:00
    Total time elapsed: 00:00:00
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Processing samples...
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Logging diagnostics...
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Optimizing policy...
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Computing loss before
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Computing KL before
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Optimizing
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Start CG optimization: #parameters: 1282, #inputs: 286, #subsample_inputs: 286
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | computing loss before
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | performing update
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | computing gradient
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | gradient computed
    2019-01-31 23:05:38 | [experiment_2019_01_31_23_05_29_0001] itr #0 | computing descent direction
    2019-01-31 23:05:39 | [experiment_2019_01_31_23_05_29_0001] itr #0 | descent direction computed
    2019-01-31 23:05:39 | [experiment_2019_01_31_23_05_29_0001] itr #0 | backtrack iters: 3
    2019-01-31 23:05:39 | [experiment_2019_01_31_23_05_29_0001] itr #0 | computing loss after
    2019-01-31 23:05:39 | [experiment_2019_01_31_23_05_29_0001] itr #0 | optimization finished
    2019-01-31 23:05:39 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Computing KL after
    2019-01-31 23:05:39 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Computing loss after
    2019-01-31 23:05:39 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Fitting baseline...
    2019-01-31 23:05:39 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Saving snapshot...
    2019-01-31 23:05:39 | [experiment_2019_01_31_23_05_29_0001] itr #0 | Saved
    2019-01-31 23:05:39 | --------------------------  -------------
    2019-01-31 23:05:39 | AverageDiscountedReturn      13.1255
    2019-01-31 23:05:39 | AverageReturn                14.1224
    2019-01-31 23:05:39 | Baseline/ExplainedVariance   -1.5755e-08
    2019-01-31 23:05:39 | Entropy                       0.579951
    2019-01-31 23:05:39 | EnvExecTime                   0.0472133
    2019-01-31 23:05:39 | Iteration                     0
    2019-01-31 23:05:39 | ItrTime                       1.71296
    2019-01-31 23:05:39 | MaxReturn                    36
    2019-01-31 23:05:39 | MinReturn                     8
    2019-01-31 23:05:39 | NumTrajs                    286
    2019-01-31 23:05:39 | Perplexity                    1.78595
    2019-01-31 23:05:39 | PolicyExecTime                0.163933
    2019-01-31 23:05:39 | ProcessExecTime               0.0250623
    2019-01-31 23:05:39 | StdReturn                     4.98905
    2019-01-31 23:05:39 | Time                          1.71285
    2019-01-31 23:05:39 | policy/Entropy                0.0648728
    2019-01-31 23:05:39 | policy/KL                     0.00501609
    2019-01-31 23:05:39 | policy/KLBefore               0
    2019-01-31 23:05:39 | policy/LossAfter             -0.00198542
    2019-01-31 23:05:39 | policy/LossBefore            -7.64309e-07
    2019-01-31 23:05:39 | policy/dLoss                  0.00198465
    2019-01-31 23:05:39 | --------------------------  -------------


Note that the execution of the experiment (including the construction of relevant objects, like environment, policy, algorithm, etc.) has been wrapped in a function call, which is then passed to the `run_experiment` method, which serializes the fucntion call, and launches a script that actually runs the experiment.

The benefit for launching experiment this way is that we separate the configuration of experiment parameters and the actual execution of the experiment. `run_experiment` supports multiple ways of running the experiment, either locally, locally in a docker container, or remotely on ec2 (see the section on :ref:`cluster`). Multiple experiments with different hyper-parameter settings can be quickly constructed and launched simultaneously on multiple ec2 machines using this abstraction.


Additional arguments for `run_experiment`:

- `exp_name`: If this is set, the experiment data will be stored in the folder `data/local/{exp_name}`. By default, the folder name is set to `experiment_{timestamp}`.
- `exp_prefix`: If this is set, and if `exp_name` is not specified, the experiment folder name will be set to `{exp_prefix}_{timestamp}`.

Running Experiments with TensorFlow and GPU
=====================

To run experiments in the TensorFlow tree of garage with the GPU enabled, set the flags use_tf and use_gpu to True when calling `run_experiment`, as shown in the code below:

.. code-block:: python

    run_experiment(
        run_task,
        # Number of parallel workers for sampling
        n_parallel=1,
        # Only keep the snapshot parameters for the last iteration
        snapshot_mode="last",
        # Specifies the seed for the experiment. If this is not provided, a random seed
        # will be used
        seed=1,
        # Always set to True when using TensorFlow
        use_tf=True,
        # Set to True to use GPU with TensorFlow
        use_gpu=True,
        # plot=True,
    )

It's also possible to run TensorFlow with only the CPU by setting use_gpu to False, which is the default behavior when use_tf is enabled.
