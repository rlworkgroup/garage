#!/usr/bin/env python3
"""This is an example to train PPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld.benchmarks as mwb
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=500)
@click.option('--batch_size', default=1024)
@wrap_experiment(snapshot_mode='all')
def mtppo_metaworld_ml1_push(ctxt, seed, epochs, batch_size):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.

    """
    set_seed(seed)
    env = normalize(
        GymEnv(mwb.ML1.get_train_tasks('push-v1'), max_episode_length=150))

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               discount=0.99,
               gae_lambda=0.95,
               center_adv=True,
               lr_clip_range=0.2)

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs, batch_size=batch_size)


mtppo_metaworld_ml1_push()
