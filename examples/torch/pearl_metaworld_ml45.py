#!/usr/bin/env python3
"""PEARL ML45 example."""

import click
import metaworld.benchmarks as mwb

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.experiment import LocalRunner
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import EnvPoolSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import PEARL
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (ContextConditionedPolicy,
                                   TanhGaussianMLPPolicy)
from garage.torch.q_functions import ContinuousMLPQFunction


@click.command()
@click.option('--num_epochs', default=1000)
@click.option('--num_train_tasks', default=45)
@click.option('--num_test_tasks', default=5)
@click.option('--encoder_hidden_size', default=200)
@click.option('--net_size', default=300)
@click.option('--num_steps_per_epoch', default=4000)
@click.option('--num_initial_steps', default=4000)
@click.option('--num_steps_prior', default=750)
@click.option('--num_extra_rl_steps_posterior', default=750)
@click.option('--batch_size', default=256)
@click.option('--embedding_batch_size', default=64)
@click.option('--embedding_mini_batch_size', default=64)
@click.option('--max_episode_length', default=150)
@wrap_experiment
def pearl_metaworld_ml45(ctxt=None,
                         seed=1,
                         num_epochs=1000,
                         num_train_tasks=45,
                         num_test_tasks=5,
                         latent_size=7,
                         encoder_hidden_size=200,
                         net_size=300,
                         meta_batch_size=16,
                         num_steps_per_epoch=4000,
                         num_initial_steps=4000,
                         num_tasks_sample=15,
                         num_steps_prior=750,
                         num_extra_rl_steps_posterior=750,
                         batch_size=256,
                         embedding_batch_size=64,
                         embedding_mini_batch_size=64,
                         max_episode_length=150,
                         reward_scale=10.,
                         use_gpu=False):
    """Train PEARL with ML45 environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        num_epochs (int): Number of training epochs.
        num_train_tasks (int): Number of tasks for training.
        num_test_tasks (int): Number of tasks for testing.
        latent_size (int): Size of latent context vector.
        encoder_hidden_size (int): Output dimension of dense layer of the
            context encoder.
        net_size (int): Output dimension of a dense layer of Q-function and
            value function.
        meta_batch_size (int): Meta batch size.
        num_steps_per_epoch (int): Number of iterations per epoch.
        num_initial_steps (int): Number of transitions obtained per task before
            training.
        num_tasks_sample (int): Number of random tasks to obtain data for each
            iteration.
        num_steps_prior (int): Number of transitions to obtain per task with
            z ~ prior.
        num_extra_rl_steps_posterior (int): Number of additional transitions
            to obtain per task with z ~ posterior that are only used to train
            the policy and NOT the encoder.
        batch_size (int): Number of transitions in RL batch.
        embedding_batch_size (int): Number of transitions in context batch.
        embedding_mini_batch_size (int): Number of transitions in mini context
            batch; should be same as embedding_batch_size for non-recurrent
            encoder.
        max_episode_length (int): Maximum episode length.
        reward_scale (int): Reward scale.
        use_gpu (bool): Whether or not to use GPU for training.

    """
    set_seed(seed)
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)
    # create multi-task environment and sample tasks
    ml45_train_envs = [
        normalize(GymEnv(mwb.ML45.from_task(task_name)))
        for task_name in mwb.ML45.get_train_tasks().all_task_names
    ]

    ml45_test_envs = [
        normalize(GymEnv(mwb.ML45.from_task(task_name)))
        for task_name in mwb.ML45.get_test_tasks().all_task_names
    ]

    env_sampler = EnvPoolSampler(ml45_train_envs)
    env_sampler.grow_pool(num_train_tasks)
    env = env_sampler.sample(num_train_tasks)
    test_env_sampler = EnvPoolSampler(ml45_test_envs)
    test_env_sampler.grow_pool(num_test_tasks)

    runner = LocalRunner(ctxt)

    # instantiate networks
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    pearl = PEARL(
        env=env,
        policy_class=ContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf=qf,
        vf=vf,
        num_train_tasks=num_train_tasks,
        num_test_tasks=num_test_tasks,
        latent_dim=latent_size,
        encoder_hidden_sizes=encoder_hidden_sizes,
        test_env_sampler=test_env_sampler,
        meta_batch_size=meta_batch_size,
        num_steps_per_epoch=num_steps_per_epoch,
        num_initial_steps=num_initial_steps,
        num_tasks_sample=num_tasks_sample,
        num_steps_prior=num_steps_prior,
        num_extra_rl_steps_posterior=num_extra_rl_steps_posterior,
        batch_size=batch_size,
        embedding_batch_size=embedding_batch_size,
        embedding_mini_batch_size=embedding_mini_batch_size,
        max_episode_length=max_episode_length,
        reward_scale=reward_scale,
    )

    set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        pearl.to()

    runner.setup(algo=pearl,
                 env=env[0](),
                 sampler_cls=LocalSampler,
                 sampler_args=dict(max_episode_length=max_episode_length),
                 n_workers=1,
                 worker_class=PEARLWorker)

    runner.train(n_epochs=num_epochs, batch_size=batch_size)


pearl_metaworld_ml45()
