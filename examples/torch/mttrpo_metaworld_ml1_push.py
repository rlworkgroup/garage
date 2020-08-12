#!/usr/bin/env python3
"""This is an example to train TRPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld.benchmarks as mwb
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.torch.algos import TRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=500)
@click.option('--batch_size', default=1024)
@wrap_experiment(snapshot_mode='all')
def mttrpo_metaworld_ml1_push(ctxt, seed, epochs, batch_size):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.

    """
    set_seed(seed)
    env = normalize(GymEnv(mwb.ML1.get_train_tasks('push-v1')))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(64, 64),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(32, 32),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = TRPO(env_spec=env.spec,
                policy=policy,
                value_function=value_function,
                max_episode_length=128,
                discount=0.99,
                gae_lambda=0.95)

    runner = LocalRunner(ctxt)
    runner.setup(algo, env)
    runner.train(n_epochs=epochs, batch_size=batch_size)


mttrpo_metaworld_ml1_push()
