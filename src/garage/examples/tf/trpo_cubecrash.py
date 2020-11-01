#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

Here it runs CubeCrash-v0 environment with 100 iterations.
"""
import click

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.tf.algos import TRPO
from garage.tf.baselines import GaussianCNNBaseline
from garage.tf.policies import CategoricalCNNPolicy
from garage.trainer import TFTrainer


@click.command()
@click.option('--batch_size', type=int, default=4000)
@click.option('--max_episode_length', type=int, default=5)
@wrap_experiment
def trpo_cubecrash(ctxt=None, seed=1, max_episode_length=5, batch_size=4000):
    """Train TRPO with CubeCrash-v0 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_episode_length (int): Maximum length of a single episode.
        batch_size (int): Number of timesteps to use in each training step.

    """
    set_seed(seed)
    with TFTrainer(ctxt) as trainer:
        env = normalize(
            GymEnv('CubeCrash-v0', max_episode_length=max_episode_length))
        policy = CategoricalCNNPolicy(env_spec=env.spec,
                                      filters=((32, (8, 8)), (64, (4, 4))),
                                      strides=(4, 2),
                                      padding='VALID',
                                      hidden_sizes=(32, 32))

        baseline = GaussianCNNBaseline(env_spec=env.spec,
                                       filters=((32, (8, 8)), (64, (4, 4))),
                                       strides=(4, 2),
                                       padding='VALID',
                                       hidden_sizes=(32, 32),
                                       use_trust_region=True)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=0.99,
                    gae_lambda=0.95,
                    lr_clip_range=0.2,
                    policy_ent_coeff=0.0)

        trainer.setup(algo, env)
        trainer.train(n_epochs=100, batch_size=batch_size)


trpo_cubecrash()
