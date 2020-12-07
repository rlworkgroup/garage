# Run Meta-/Multi-Task RL Experiments

This guide will walk you through how to run meta-/ multi-task RL experiments in garage.

Similar to running all other experiments in garage, running meta-/ multi-task RL experiments generally involves steps such as:

- Defining the experiement with the `wrap_experiment` decorator
- Constructing a `Trainer`
- Constructing an environment
- Constructing policy/ algorithm object(s)

In meta-/multi-task RL experiment, it revolves around solving multiple tasks and hence the construction of multiple/specific environment(s). Belows are a few environment wrappers commonly used in garage for meta-/multi-task RL learning:

- `MultiEnvWrapper`: a wrapper for mulitiple environments
- `RL2Env`: a specific wrapper for RL2 environment

Also, it's worth noting that [MetaWorld](https://github.com/rlworkgroup/metaworld) provides a variety of benchmarked robotics tasks and can be extensively used in garage.

## Meta-RL experiments

The garage repository contains several meta-RL experiment examples. We will take a look at `te_ppo_metaworld_ml1_push.py` as below:

```py
import metaworld
import tensorflow as tf

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper
from garage.experiment import TFTrainer
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearMultiFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import TEPPO
from garage.tf.algos.te import TaskEmbeddingWorker
from garage.tf.embeddings import GaussianMLPEncoder
from garage.tf.policies import GaussianMLPTaskEmbeddingPolicy

@wrap_experiment
def te_ppo_ml1_push(ctxt, seed, n_epochs, batch_size_per_task):
    """Train Task Embedding PPO with PointEnv.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        n_epochs (int): Total number of epochs for training.
        batch_size_per_task (int): Batch size of samples for each task.

    """
    set_seed(seed)
    n_tasks = 50
    mt1 = metaworld.MT1('push-v1')
    task_sampler = MetaWorldTaskSampler(mt1,
                                        'train',
                                        lambda env, _: normalize(env),
                                        add_env_onehot=False)
    envs = [env_up() for env_up in task_sampler.sample(n_tasks)]
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')

    latent_length = 2
    inference_window = 6
    batch_size = batch_size_per_task
    policy_ent_coeff = 2e-2
    encoder_ent_coeff = 2e-4
    inference_ce_coeff = 5e-2
    max_episode_length = 100
    embedding_init_std = 0.1
    embedding_max_std = 0.2
    embedding_min_std = 1e-6
    policy_init_std = 1.0
    policy_max_std = None
    policy_min_std = None

    with TFTrainer(snapshot_config=ctxt) as trainer:

        task_embed_spec = TEPPO.get_encoder_spec(env.task_space,
                                                 latent_dim=latent_length)

        task_encoder = GaussianMLPEncoder(
            name='embedding',
            embedding_spec=task_embed_spec,
            hidden_sizes=(20, 20),
            std_share_network=True,
            init_std=embedding_init_std,
            max_std=embedding_max_std,
            output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        traj_embed_spec = TEPPO.get_infer_spec(
            env.spec,
            latent_dim=latent_length,
            inference_window_size=inference_window)

        inference = GaussianMLPEncoder(
            name='inference',
            embedding_spec=traj_embed_spec,
            hidden_sizes=(20, 10),
            std_share_network=True,
            init_std=2.0,
            output_nonlinearity=tf.nn.tanh,
            min_std=embedding_min_std,
        )

        policy = GaussianMLPTaskEmbeddingPolicy(
            name='policy',
            env_spec=env.spec,
            encoder=task_encoder,
            hidden_sizes=(32, 16),
            std_share_network=True,
            max_std=policy_max_std,
            init_std=policy_init_std,
            min_std=policy_min_std,
        )

        baseline = LinearMultiFeatureBaseline(
            env_spec=env.spec, features=['observations', 'tasks', 'latents'])

        sampler = LocalSampler(agents=policy,
                               envs=env,
                               max_episode_length=env.spec.max_episode_length,
                               is_tf_worker=True,
                               worker_class=TaskEmbeddingWorker)

        algo = TEPPO(env_spec=env.spec,
                     policy=policy,
                     baseline=baseline,
                     inference=inference,
                     max_episode_length=max_episode_length,
                     discount=0.99,
                     lr_clip_range=0.2,
                     policy_ent_coeff=policy_ent_coeff,
                     encoder_ent_coeff=encoder_ent_coeff,
                     inference_ce_coeff=inference_ce_coeff,
                     use_softplus_entropy=True,
                     optimizer_args=dict(
                         batch_size=32,
                         max_epochs=10,
                         learning_rate=1e-3,
                     ),
                     inference_optimizer_args=dict(
                         batch_size=32,
                         max_epochs=10,
                     ),
                     center_adv=True,
                     stop_ce_gradient=True)

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size, plot=False)


te_ppo_ml1_push()
```

Note that `envs` are a list of `env` created from `ML1-reach-v1` environment in `metaworld.benchmarks`.

To handle multiple environments, `envs` object can be passed to `MultiEnvWrapper` along with several other arguments such as `sample_strategy`, `mode`, `env_names`.

In this example, we assume one-hot task id is appened to observation and to exclude that, the call to `MultiEnvWrapper(envs)` is replaced with `MultiEnvWrapper(envs, mode='del-onehot')`.

## Multi-task RL experiments

When performing a multi-task RL experiment, we can use multi-task learning environment such as `MT50`, `MT10` etc. We will take a look at `te_ppo_metaworld_mt50.py` as below:

```eval_rst
.. literalinclude:: ../../examples/tf/te_ppo_metaworld_mt50.py
```

In this example, to sample tasks from `MT50` environments in a round robin fashion, the call to `MultiEnvWrapper` becomes `MultiEnvWrapper(envs , sample_strategy=round_robin_strategy, mode='del-onehot')`.

## Garage's meta-/ multi-task RL benchmark

Garage benchmarks the following meta-/ multi RL experiements:

```eval_rst
+---------------+-------------+------------+--------------------+
| Algorithm     | Observation | Action     | Environment Set    |
+===============+=============+============+====================+
| Meta-RL       | Non-Pixel   | Discrete   | *ML_ENV_SET        |
+---------------+-------------+------------+--------------------+
| Multi-Task RL | Non-Pixel   | Discrete   | *MT_ENV_SET        |
+---------------+-------------+------------+--------------------+
```

```py
*ML_ENV_SET = ['ML1-push-v1' ,  'ML1-reach-v1' ,  'ML1-pick-place-v1' ,  'ML10' ,  'ML45']
*MT_ENV_SET = ['ML1-push-v1' ,  'ML1-reach-v1' ,  'ML1-pick-place-v1' ,  'MT10' ,  'MT50' ]
```

See [`docs/benchmarking.md`](https://garage.readthedocs.io/en/latest/user/benchmarking.html) for a more detailed explaination on garage's benchmarking.

----

*This page was authored by Iris Liu ([@irisliucy](https://github.com/irisliucy)).*
