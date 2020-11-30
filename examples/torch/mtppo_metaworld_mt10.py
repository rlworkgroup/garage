#!/usr/bin/env python3
"""This is an example to train PPO on MT10 environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld
import psutil
import torch

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment import MetaWorldTaskSampler
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=500)
@click.option('--batch_size', default=1024)
@click.option('--n_workers', default=psutil.cpu_count(logical=False))
@click.option('--n_tasks', default=10)
@wrap_experiment(snapshot_mode='all')
def mtppo_metaworld_mt10(ctxt, seed, epochs, batch_size, n_workers, n_tasks):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        batch_size (int): Number of environment steps in one batch.
        n_workers (int): The number of workers the sampler should use.
        n_tasks (int): Number of tasks to use. Should be a multiple of 10.

    """
    set_seed(seed)
    mt10 = metaworld.MT10()
    train_task_sampler = MetaWorldTaskSampler(mt10,
                                              'train',
                                              lambda env, _: normalize(env),
                                              add_env_onehot=True)
    assert n_tasks % 10 == 0
    assert n_tasks <= 500
    envs = [env_up() for env_up in train_task_sampler.sample(n_tasks)]
    env = MultiEnvWrapper(envs,
                          sample_strategy=round_robin_strategy,
                          mode='vanilla')

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

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=n_workers)

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.95,
               center_adv=True,
               lr_clip_range=0.2)

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs, batch_size=batch_size)


mtppo_metaworld_mt10()
