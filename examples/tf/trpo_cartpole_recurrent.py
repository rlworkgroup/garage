#!/usr/bin/env python3
"""This is an example to train a task with TRPO algorithm.

It uses an LSTM-based recurrent policy.

Here it runs CartPole-v1 environment with 100 iterations.

Results:
    AverageReturn: 100
    RiseTime: itr 13
"""
# pylint: disable=no-value-for-parameter
import click

from garage import wrap_experiment
from garage.envs import GymEnv
from garage.experiment.deterministic import set_seed
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import RaySampler
from garage.tf.algos import TRPO
from garage.tf.optimizers import (ConjugateGradientOptimizer,
                                  FiniteDifferenceHVP)
from garage.tf.policies import CategoricalLSTMPolicy
from garage.trainer import TFTrainer


@click.command()
@click.option('--seed', default=1)
@click.option('--n_epochs', default=100)
@click.option('--batch_size', default=4000)
@click.option('--plot', default=False)
@wrap_experiment
def trpo_cartpole_recurrent(ctxt, seed, n_epochs, batch_size, plot):
    """Train TRPO with a recurrent policy on CartPole.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        n_epochs (int): Number of epochs for training.
        seed (int): Used to seed the random number generator to produce
            determinism.
        batch_size (int): Batch size used for training.
        plot (bool): Whether to plot or not.

    """
    set_seed(seed)
    with TFTrainer(snapshot_config=ctxt) as trainer:
        env = GymEnv('CartPole-v1', max_episode_length=100)

        policy = CategoricalLSTMPolicy(name='policy', env_spec=env.spec)

        baseline = LinearFeatureBaseline(env_spec=env.spec)

        sampler = RaySampler(agents=policy,
                             envs=env,
                             max_episode_length=env.spec.max_episode_length,
                             is_tf_worker=True)

        algo = TRPO(env_spec=env.spec,
                    policy=policy,
                    baseline=baseline,
                    sampler=sampler,
                    discount=0.99,
                    max_kl_step=0.01,
                    optimizer=ConjugateGradientOptimizer,
                    optimizer_args=dict(hvp_approach=FiniteDifferenceHVP(
                        base_eps=1e-5)))

        trainer.setup(algo, env)
        trainer.train(n_epochs=n_epochs, batch_size=batch_size, plot=plot)


trpo_cartpole_recurrent()
