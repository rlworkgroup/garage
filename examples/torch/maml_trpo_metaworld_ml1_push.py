#!/usr/bin/env python3
"""This is an example to train MAML-TRPO on ML1 Push environment."""
# pylint: disable=no-value-for-parameter
import click
import metaworld.benchmarks as mwb
import torch

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import LocalRunner, MetaEvaluator
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.torch.algos import MAMLTRPO
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.value_functions import GaussianMLPValueFunction


@click.command()
@click.option('--seed', default=1)
@click.option('--epochs', default=300)
@click.option('--episodes_per_task', default=10)
@click.option('--meta_batch_size', default=20)
@wrap_experiment(snapshot_mode='all')
def maml_trpo_metaworld_ml1_push(ctxt, seed, epochs, episodes_per_task,
                                 meta_batch_size):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        epochs (int): Number of training epochs.
        episodes_per_task (int): Number of episodes per epoch per task
            for training.
        meta_batch_size (int): Number of tasks sampled per batch.

    """
    set_seed(seed)
    env = normalize(GymEnv(mwb.ML1.get_train_tasks('push-v1')),
                    expected_action_scale=10.)

    policy = GaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=(100, 100),
        hidden_nonlinearity=torch.tanh,
        output_nonlinearity=None,
    )

    value_function = GaussianMLPValueFunction(env_spec=env.spec,
                                              hidden_sizes=[32, 32],
                                              hidden_nonlinearity=torch.tanh,
                                              output_nonlinearity=None)

    max_episode_length = 100

    test_sampler = SetTaskSampler(lambda: normalize(
        GymEnv(mwb.ML1.get_test_tasks('push-v1'))))

    meta_evaluator = MetaEvaluator(test_task_sampler=test_sampler,
                                   max_episode_length=max_episode_length)

    runner = LocalRunner(ctxt)
    algo = MAMLTRPO(env=env,
                    policy=policy,
                    value_function=value_function,
                    max_episode_length=max_episode_length,
                    meta_batch_size=meta_batch_size,
                    discount=0.99,
                    gae_lambda=1.,
                    inner_lr=0.1,
                    num_grad_updates=1,
                    meta_evaluator=meta_evaluator)

    runner.setup(algo, env)
    runner.train(n_epochs=epochs,
                 batch_size=episodes_per_task * max_episode_length)


maml_trpo_metaworld_ml1_push()
