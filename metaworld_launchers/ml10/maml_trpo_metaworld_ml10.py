#!/usr/bin/env python3
"""This is an example to train MAML-TRPO on ML10 environment."""
# pylint: disable=no-value-for-parameter
# yapf: disable
import click
import torch

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv
from garage.experiment import (MetaEvaluator, MetaWorldTaskSampler,
                               SetTaskSampler)
from garage.experiment.deterministic import set_seed
from garage.sampler import RaySampler, LocalSampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction
from garage.trainer import Trainer

# yapf: enable
# import pip
# package = f'metaworld @ https://git@api.github.com/repos/rlworkgroup/metaworld/tarball/new-reward-functions'
# pip.main(['uninstall', '--yes', package])
# pip.main(['install', package])
import metaworld

@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=4000)
@click.option('--episodes_per_task', default=10)
@click.option('--meta_batch_size', default=20)
@wrap_experiment(snapshot_mode='gap', snapshot_gap=16, name_parameters='passed')
def maml_trpo_metaworld_ml10(ctxt, seed, epochs, episodes_per_task,
                             meta_batch_size):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (ExperimentContext): The experiment configuration used by
            :class:`~Trainer: to create the :class:`~Snapshotter:.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_task (int): Number of episodes per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.

    """
    set_seed(seed)
    ml10 = metaworld.ML10()
    tasks = MetaWorldTaskSampler(ml10, 'train')
    env = tasks.sample(10)[0]()
    test_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                  env=MetaWorldSetTaskEnv(ml10, 'test'))
    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(128, 128),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=(64, 64),
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)
    num_test_envs = 5
    meta_evaluator = MetaEvaluator(test_task_sampler=test_sampler,
                                   n_exploration_eps=episodes_per_task,
                                   n_test_tasks=num_test_envs * 2,
                                   n_test_episodes=10)

    sampler = RaySampler(agents=policy,
                         envs=env,
                         max_episode_length=env.spec.max_episode_length,
                         n_workers=meta_batch_size)

    trainer = Trainer(ctxt)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    sampler=sampler,
                    task_sampler=tasks,
                    value_function=value_function,
                    meta_batch_size=meta_batch_size,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=0.1,
                    num_grad_updates=1,
                    meta_evaluator=meta_evaluator,
                    evaluate_every_n_epochs=8)

    trainer.setup(algo, env)
    trainer.train(n_epochs=epochs,
                  batch_size=episodes_per_task * env.spec.max_episode_length)


maml_trpo_metaworld_ml10()
