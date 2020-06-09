.. _experiments:


===================
Running Experiments
===================

In garage, experiments are described using Python files we call "experiment
launchers." There is nothing unusual about how experiment launchers are
evaluated, and we recommend making use of off-the-shelf python libraries for
common tasks such as command line argument parsing, experiment configuration,
or remote execution.

All experiment launchers eventually call a function wrapped with a decorator
called :code:`wrap_experiment`, which defines the scope of an experiment, and
handles common tasks like setting up a log directory for the results of the
experiment.

Within the decorated experiment function, experiment launchers then construct
the important objects involved in running an experiment, such as the following:

 - The :code:`LocalRunner`, which sets up important state (such as a TensorFlow Session) for running the algorithm in the experiment.
 - The :code:`environment` object, which is the environment in which reinforcement learning is being done.
 - The :code:`policy` object, which is trained to optimize for maximal reward in the :code:`environment`.
 - The :code:`algorithm`, which trains the :code:`policy`.

Finally, the launcher calls :code:`runner.setup` and :code:`runner.train` which co-ordinate running the algorithm.

The garage repository contains several example experiment launchers. A fairly
simple one, :code:`examples/tf/trpo_cartpole.py`, is also pasted below:

.. code-block:: python

  from garage import wrap_experiment
  from garage.envs import GarageEnv
  from garage.experiment import LocalTFRunner
  from garage.experiment.deterministic import set_seed
  from garage.np.baselines import LinearFeatureBaseline
  from garage.tf.algos import TRPO
  from garage.tf.policies import CategoricalMLPPolicy


  @wrap_experiment
  def trpo_cartpole(ctxt=None, seed=1):
      """Train TRPO with CartPole-v1 environment.

      Args:
          ctxt (garage.experiment.ExperimentContext): The experiment
              configuration used by LocalRunner to create the snapshotter.
          seed (int): Used to seed the random number generator to produce
              determinism.

      """
      set_seed(seed)
      with LocalTFRunner(ctxt) as runner:
          env = GarageEnv(env_name='CartPole-v1')

          policy = CategoricalMLPPolicy(name='policy',
                                        env_spec=env.spec,
                                        hidden_sizes=(32, 32))

          baseline = LinearFeatureBaseline(env_spec=env.spec)

          algo = TRPO(env_spec=env.spec,
                      policy=policy,
                      baseline=baseline,
                      max_path_length=100,
                      discount=0.99,
                      max_kl_step=0.01)

          runner.setup(algo, env)
          runner.train(n_epochs=100, batch_size=4000)


  trpo_cartpole()


Running the above should produce output like:

.. code-block:: text

  ...

  2020-05-11 14:13:05 | [trpo_cartpole] Logging to /home/kr/garage/data/local/experiment/trpo_cartpole_1
  2020-05-11 14:13:05 | [trpo_cartpole] Setting seed to 1
  2020-05-11 14:13:06 | [trpo_cartpole] Obtaining samples...
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | Obtaining samples for iteration 0...
  0% [##############################] 100% | ETA: 00:00:00
  Total time elapsed: 00:00:00
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | Logging diagnostics...
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | Optimizing policy...
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | Computing loss before
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | Computing KL before
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | Optimizing
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | Start CG optimization: #parameters: 1282, #inputs: 201, #subsample_inputs: 201
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | computing loss before
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | computing gradient
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | gradient computed
  2020-05-11 14:13:06 | [trpo_cartpole] epoch #0 | computing descent direction
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | descent direction computed
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | backtrack iters: 4
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | optimization finished
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | Computing KL after
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | Computing loss after
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | Fitting baseline...
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | Saving snapshot...
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | Saved
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | Time 1.25 s
  2020-05-11 14:13:07 | [trpo_cartpole] epoch #0 | EpochTime 1.25 s
  ---------------------------------------  --------------
  Entropy                                     0.690996
  EnvExecTime                                 0.0628054
  Evaluation/AverageDiscountedReturn         17.8993
  Evaluation/AverageReturn                   20.1095
  Evaluation/CompletionRate                   1
  Evaluation/Iteration                        0
  Evaluation/MaxReturn                       61
  Evaluation/MinReturn                        9
  Evaluation/NumTrajs                       201
  Evaluation/StdReturn                       10.0935
  Extras/EpisodeRewardMean                   20.43
  LinearFeatureBaseline/ExplainedVariance    -2.65605e-08
  Perplexity                                  1.9957
  PolicyExecTime                              0.430455
  ProcessExecTime                             0.0215859
  TotalEnvSteps                            4042
  policy/Entropy                              0.687919
  policy/KL                                   0.0051155
  policy/KLBefore                             0
  policy/LossAfter                           -0.0077831
  policy/LossBefore                          -3.77624e-07
  policy/dLoss                                0.00778273
  ---------------------------------------  --------------


Note that the :code:`wrap_experiment` wrapped function still acts like a normal function, but requires all arguments to be passed by keyword. The function will automatically allocate an experiment directory based on the name of the wrapped function, and save various files to assist in reproducing the experiment (such as all of the arguments to the wrapped function).

Several arguments can be passed to :code:`wrap_experiment`, or passed as a dictionary as the first argument to the wrapped function.

For example, to use a specific log directory, the call to :code:`trpo_cartpole()` above can be replaced with :code:`trpo_cartpole({log_dir: 'my/log/directory', use_existing_dir: True}, seed=100)`.

For additional details on the other objects used in experiment launchers, we recommend browsing the reference documentation, or using Python's dynamic documentation tools.

For example:

.. code-block:: text

  >>> print(garage.wrap_experiment.__doc__)
  Decorate a function to turn it into an ExperimentTemplate.

      When invoked, the wrapped function will receive an ExperimentContext, which
      will contain the log directory into which the experiment should log
      information.

      This decorator can be invoked in two differed ways.

      Without arguments, like this:

          @wrap_experiment
          def my_experiment(ctxt, seed, lr=0.5):
              ...

      Or with arguments:

          @wrap_experiment(snapshot_mode='all')
          def my_experiment(ctxt, seed, lr=0.5):
              ...

      All arguments must be keyword arguments.

      Args:
          function (callable or None): The experiment function to wrap.
          log_dir (str or None): The full log directory to log to. Will be
              computed from `name` if omitted.
          name (str or None): The name of this experiment template. Will be
              filled from the wrapped function's name if omitted.
          prefix (str): Directory under data/local in which to place the
              experiment directory.
          snapshot_mode (str): Policy for which snapshots to keep (or make at
              all). Can be either "all" (all iterations will be saved), "last"
              (only the last iteration will be saved), "gap" (every snapshot_gap
              iterations are saved), or "none" (do not save snapshots).
          snapshot_gap (int): Gap between snapshot iterations. Waits this number
              of iterations before taking another snapshot.
          archive_launch_repo (bool): Whether to save an archive of the
              repository containing the launcher script. This is a potentially
              expensive operation which is useful for ensuring reproducibility.
          name_parameters (str or None): Parameters to insert into the experiment
              name. Should be either None (the default), 'all' (all parameters
              will be used), or 'passed' (only passed parameters will be used).
              The used parameters will be inserted in the order they appear in
              the function definition.
          use_existing_dir (bool): If true, (re)use the directory for this
              experiment, even if it already contains data.

      Returns:
          callable: The wrapped function.

Running Experiments on GPU / CPU
================================

When training on-policy RL algorithms (such as PPO and TRPO) on a low-dimensional (i.e. non-image) environment using a GPU typically results in `slower` training overall.
However, TensorFlow will default to using a GPU if one is available. This can be changed by setting the :code:`CUDA_VISIBLE_DEVICES` environment variable.

.. code-block:: bash

  export CUDA_VISIBLE_DEVICES=-1  # CPU only
  python path/to/my/experiment/launcher.py

When training off-policy RL algorithms (such as DDPG, TD3, SAC, and PEARL), using a GPU generally allows faster training.
However, PyTorch won't use a GPU by default.

In order to enable the GPU for PyTorch, add the following code snippets to the experiment launcher.

.. code-block:: python

  import torch
  import garage.torch.utils as tu

  ...

    if torch.cuda.is_available():
        tu.set_gpu_mode(True)
    else:
        tu.set_gpu_mode(False)
    algo.to()

See :code:`examples/torch/sac_half_cheetah_batch.py` for a more detailed example.
