#!/usr/bin/env python3
"""Example script to run RL2 in HalfCheetah."""
# pylint: disable=no-value-for-parameter
import click

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.envs.mujoco.half_cheetah_vel_env import HalfCheetahVelEnv
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
@click.option('--max_episode_length', default=100)
@click.option('--meta_batch_size', default=10)
@click.option('--n_epochs', default=10)
@click.option('--episode_per_task', default=4)
@wrap_experiment
def rl2_ppo_halfcheetah(ctxt, seed, max_episode_length, meta_batch_size,
                        n_epochs, episode_per_task):
    """Train PPO with HalfCheetah environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_episode_length (int): Maximum length of a single episode.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    trainer = Trainer(snapshot_config=ctxt)

    inner_max_episode_length = max_episode_length * episode_per_task
    tasks = task_sampler.SetTaskSampler(lambda: RL2Env(
        GymEnv(HalfCheetahVelEnv())))

    env_spec = RL2Env(
        GymEnv(HalfCheetahVelEnv(),
               max_episode_length=inner_max_episode_length)).spec
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
                  episodes_per_trial=episode_per_task,
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
                  center_adv=False)

    trainer.setup(algo,
                  tasks.sample(meta_batch_size),
                  sampler_cls=LocalSampler,
                  n_workers=meta_batch_size,
                  worker_class=RL2Worker,
                  worker_args=dict(n_episodes_per_trial=episode_per_task))

    trainer.train(n_epochs=n_epochs,
                  batch_size=episode_per_task * max_episode_length *
                  meta_batch_size)


rl2_ppo_halfcheetah()
