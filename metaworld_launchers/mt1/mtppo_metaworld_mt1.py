#!/usr/bin/env python3
"""This is an example to train PPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld
import torch

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.multi_env_wrapper import MultiEnvWrapper, round_robin_strategy
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.sampler import RaySampler
from garage.torch.algos import PPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer


@click.command()
@click.option('--env-name', type=str)
@click.option('--seed', default=1)
@click.option('--epochs', default=500)
@click.option('--episodes_per_env', default=1)
@wrap_experiment(snapshot_mode='gap', snapshot_gap=50, name_parameters='passed')
def mtppo_metaworld_mt1(ctxt, env_name, seed, epochs, episodes_per_env):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_env (int): Number of episodes to sample per environment
            during each training epoch

    """
    set_seed(seed)
    n_tasks = 50
    mt1 = metaworld.MT1(env_name)
    train_task_sampler = MetaWorldTaskSampler(mt1, 'train',
                                              lambda env, _: normalize(env),
                                              add_env_onehot=True)
    envs = [env_up() for env_up in train_task_sampler.sample(n_tasks)]
    env = envs[0]

    policy = GaussianMLPPolicy(
        hidden_sizes=(256,256),
        env_spec=env.spec,
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(256,256),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    sampler = RaySampler(agents=policy,
                         envs=envs,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=len(envs))

    algo = PPO(env_spec=env.spec,
               policy=policy,
               value_function=value_function,
               sampler=sampler,
               discount=0.99,
               gae_lambda=0.95,
               center_adv=False,
               lr_clip_range=0.2,
               stop_entropy_gradient=True,
               entropy_method='max',
               policy_ent_coeff=0.02,
               )

    trainer = Trainer(ctxt)
    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs, batch_size=episodes_per_env*len(envs)*env.max_path_length)


mtppo_metaworld_mt1()
