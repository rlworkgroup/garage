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
from garage.tf.algos import RL2TRPO
from garage.tf.algos.rl2 import RL2Env, RL2Worker
from garage.tf.optimizers import (ConjugateGradientOptimizer,
                                  FiniteDifferenceHVP)
from garage.tf.policies import GaussianGRUPolicy
from garage.trainer import TFTrainer


@click.command()
@click.option('--seed', default=1)
@click.option('--max_episode_length', default=150)
@click.option('--meta_batch_size', default=10)
@click.option('--n_epochs', default=10)
@click.option('--episode_per_task', default=4)
@wrap_experiment
def rl2_trpo_halfcheetah(ctxt, seed, max_episode_length, meta_batch_size,
                         n_epochs, episode_per_task):
    """Train TRPO with HalfCheetah environment.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer` to create the :class:`~Snapshotter`.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_episode_length (int): Maximum length of a single episode.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.


    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        tasks = task_sampler.SetTaskSampler(
            HalfCheetahVelEnv,
            wrapper=lambda env, _: RL2Env(
                GymEnv(env, max_episode_length=max_episode_length)))

        env_spec = RL2Env(
            GymEnv(HalfCheetahVelEnv(),
                   max_episode_length=max_episode_length)).spec
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=env_spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        envs = tasks.sample(meta_batch_size)
        sampler = LocalSampler(
            agents=policy,
            envs=envs,
            max_episode_length=env_spec.max_episode_length,
            is_tf_worker=True,
            n_workers=meta_batch_size,
            worker_class=RL2Worker,
            worker_args=dict(n_episodes_per_trial=episode_per_task))

        algo = RL2TRPO(meta_batch_size=meta_batch_size,
                       task_sampler=tasks,
                       env_spec=env_spec,
                       policy=policy,
                       baseline=baseline,
                       sampler=sampler,
                       episodes_per_trial=episode_per_task,
                       discount=0.99,
                       max_kl_step=0.01,
                       optimizer=ConjugateGradientOptimizer,
                       optimizer_args=dict(hvp_approach=FiniteDifferenceHVP(
                           base_eps=1e-5)))

        trainer.setup(algo, envs)

        trainer.train(n_epochs=n_epochs,
                      batch_size=episode_per_task * max_episode_length *
                      meta_batch_size)


rl2_trpo_halfcheetah()
