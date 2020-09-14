#!/usr/bin/env python3
"""Example script to run RL2 in ML45."""
# pylint: disable=no-value-for-parameter
import click
import metaworld.benchmarks as mwb

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment import task_sampler
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import Trainer


@click.command()
@click.option('--seed', default=1)
@click.option('--meta_batch_size', default=50)
@click.option('--n_epochs', default=10)
@click.option('--episode_per_task', default=10)
@wrap_experiment
def rl2_ppo_metaworld_ml45(ctxt, seed, meta_batch_size, n_epochs,
                           episode_per_task):
    """Train PPO with ML45 environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)

    max_episode_length = 150
    inner_max_episode_length = max_episode_length * episode_per_task
    ml45_train_tasks = mwb.ML45.get_train_tasks()
    ml45_train_envs = [
        RL2Env(
            GymEnv(mwb.ML45.from_task(task_name),
                   max_episode_length=inner_max_episode_length))
        for task_name in ml45_train_tasks.all_task_names
    ]
    tasks = task_sampler.EnvPoolSampler(ml45_train_envs)
    tasks.grow_pool(meta_batch_size)

    env_spec = ml45_train_envs[0].spec

    policy = GaussianGRUPolicy(name='policy',
                               hidden_dim=64,
                               env_spec=env_spec,
                               state_include_action=False)

    baseline = LinearFeatureBaseline(env_spec=env_spec)

    algo = RL2PPO(meta_batch_size=meta_batch_size,
                  task_sampler=tasks,
                  env_spec=env_spec,
                  policy=policy,
                  baseline=baseline,
                  discount=0.99,
                  gae_lambda=0.95,
                  lr_clip_range=0.2,
                  optimizer_args=dict(
                      batch_size=32,
                      max_optimization_epochs=10,
                  ),
                  stop_entropy_gradient=True,
                  entropy_method='max',
                  policy_ent_coeff=0.02,
                  center_adv=False,
                  episodes_per_trial=episode_per_task)

    trainer.setup(algo,
                  tasks.sample(meta_batch_size),
                  sampler_cls=LocalSampler,
                  n_workers=meta_batch_size,
                  worker_class=RL2Worker,
                  worker_args=dict(n_episodes_per_trial=episode_per_task))

    trainer.train(n_epochs=n_epochs,
                  batch_size=episode_per_task * max_episode_length *
                  meta_batch_size)


rl2_ppo_metaworld_ml45()
