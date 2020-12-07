# Train a Policy to Solve an Environment

This page will guide you how to train a policy to solve an environment.

## Define the Experiment

In garage, we train a policy in an experiment, which is a function wrapped by a
decorator called `wrap_experiment`. Below is an simple example.
`wrap_experiment` could have some arguments. You can see the [experiments doc](experiments)
for details of running experiments.

```py
@wrap_experiment
def my_first_experiment():
    ...
```

### Construct a Trainer

Within the experiment, we need a `Trainer` to set up important state (such
as a TensorFlow Session) for training a policy. To construct a `Trainer`, an
experiment context called `ctxt` is needed. This is used to create the
snapshotter, and we can set it `None` here to make it simple.

Garage supports both PyTorch and TensorFlow. If you use TensorFlow, you should
use `TFTrainer`.

Besides, in order to produce determinism, you can set a seed for the random
number generator.

```py
@wrap_experiment
def my_first_experiment(ctxt=None, seed=1):
    set_seed(seed)
    # PyTorch
    trainer = Trainer(ctxt)
    ...
    # TensorFlow
    with TFTrainer(ctxt) as trainer:
        ...
```

### Construct an Environment

Garage supports many environments. You can also implement your own environment
like [this](implement_env). In this example, we choose `CartPole-V1`
environment.

```py
env = GymEnv('CartPole-v1')
```

### Construct a Policy and an Algorithm

Construct your policy and choose an algorithm to train it. Here, we use
`CategoricalMLPPolicy` and `TRPO`, you can also implement your own algorithm
like [this](implement_algo). The policy should be compatible with the
environment's observations and action space (CNN for image observations,
discrete policy for discrete action spaces, etc). The action space of
`CartPole-V1` is discrete so we choose a discrete policy here. Besides, as an
on policy algorithm, we need a sampler to make samples. Here we use the basic
`LocalSampler`.

```py
policy = CategoricalMLPPolicy(name='policy',
                              env_spec=env.spec,
                              hidden_sizes=(32, 32))

baseline = LinearFeatureBaseline(env_spec=env.spec)

sampler = LocalSampler(agents=policy,
                       envs=env,
                       max_episode_length=env.spec.max_episode_length,
                       is_tf_worker=True)

algo = TRPO(env_spec=env.spec,
            policy=policy,
            baseline=baseline,
            discount=0.99,
            max_kl_step=0.01)
```

### Tell the Trainer How to Train the Policy

The final step is calling `trainer.setup` and `trainer.train` to co-ordinate
training the policy.

```py
trainer.setup(algo, env)
trainer.train(n_epochs=100, batch_size=4000)
```

## Run the Experiment

To run the experiment, simply call the experiment function you just defined.

```py
my_first_experiment()
my_first_experiment(seed=3)  # changes the seed to 3
```

Usually these will appear at the end of your launcher script, but your
experiment functions are regular Python functions, and can be imported
anywhere.

See below for a full example.

## Example: Train TRPO to Solve `CartPole-v1`

In the above steps, we construct the required components to train a
`CategoricalMLPPolicy` with `TRPO` to solve `CartPole-v1` and wrap all into an
experiment function. You can find the full example in [`examples/tf/trpo_cartpole.py`](https://github.com/rlworkgroup/garage/blob/master/examples/tf/trpo_cartpole.py),
which is also pasted below:

```eval_rst
.. literalinclude:: ../../examples/tf/trpo_cartpole.py
```

Running the above should produce output like:

```sh
2020-06-25 14:03:46 | [trpo_cartpole] Logging to /home/ruofu/garage/data/local/experiment/trpo_cartpole_4
2020-06-25 14:03:48 | [trpo_cartpole] Obtaining samples...
Sampling  [####################################]  100%
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | Logging diagnostics...
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | Optimizing policy...
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | Computing loss before
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | Computing KL before
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | Optimizing
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | Start CG optimization:
#parameters: 1282, #inputs: 186, #subsample_inputs: 186
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | computing loss before
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | computing gradient
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | gradient computed
2020-06-25 14:03:52 | [trpo_cartpole] epoch #0 | computing descent direction
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | descent direction computed
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | backtrack iters: 10
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | optimization finished
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | Computing KL after
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | Computing loss after
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | Fitting baseline...
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | Saving snapshot...
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | Saved
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | Time 4.66 s
2020-06-25 14:03:53 | [trpo_cartpole] epoch #0 | EpochTime 4.66 s
---------------------------------------  --------------
Evaluation/AverageDiscountedReturn         19.045
Evaluation/AverageReturn                   21.5054
Evaluation/TerminationRate                  1
Evaluation/Iteration                        0
Evaluation/MaxReturn                       58
Evaluation/MinReturn                        8
Evaluation/NumEpisodes                    186
Evaluation/StdReturn                       10.0511
Extras/EpisodeRewardMean                   22.22
LinearFeatureBaseline/ExplainedVariance     4.14581e-08
TotalEnvSteps                            4000
policy/Entropy                              3.22253
policy/KL                                   9.75289e-05
policy/KLBefore                             0
policy/LossAfter                           -0.5136
policy/LossBefore                          -0.513123
policy/Perplexity                          25.0916
policy/dLoss                                0.000476599
---------------------------------------  --------------
```

----

*This page was authored by Ruofu Wang ([@yeukfu](https://github.com/yeukfu)).*
