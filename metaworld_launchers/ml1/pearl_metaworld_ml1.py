#!/usr/bin/env python3
"""PEARL ML10 example."""

import click
import metaworld

from garage import wrap_experiment
from garage.envs import MetaWorldSetTaskEnv, normalize
from garage.experiment.deterministic import set_seed
from garage.experiment.task_sampler import SetTaskSampler
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import PEARL
from garage.torch.algos.pearl import PEARLWorker
from garage.torch.embeddings import MLPEncoder
from garage.torch.policies import (ContextConditionedPolicy,
                                   TanhGaussianMLPPolicy)
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer


@click.command()
@click.option('--env_name', type=str, default='pick-place-v2')
@click.option('--seed', default=1, type=int)
@wrap_experiment(snapshot_mode='none', name_parameters='passed')
def pearl_metaworld_ml1(ctxt,
                         env_name,
                         seed=1,
                         num_epochs=1000,
                         latent_size=7,
                         encoder_hidden_size=200,
                         net_size=300,
                         num_initial_steps=15000,
                         num_steps_prior=2500,
                         num_extra_rl_steps_posterior=2500,
                         batch_size=1000,
                         embedding_batch_size=250,
                         embedding_mini_batch_size=250,
                         reward_scale=1000.,
                         use_gpu=True):
    """Train PEARL with ML10 environments.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        num_epochs (int): Number of training epochs.
        num_train_tasks (int): Number of tasks for training.
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
        reward_scale (int): Reward scale.
        use_gpu (bool): Whether or not to use GPU for training.

    """
    num_steps_per_epoch = 40
    num_tasks_sample = num_train_tasks = 15
    meta_batch_size = 16

    set_seed(seed)
    encoder_hidden_sizes = (encoder_hidden_size, encoder_hidden_size,
                            encoder_hidden_size)
    # create multi-task environment and sample tasks
    ml1 = metaworld.ML1(env_name)
    train_env = MetaWorldSetTaskEnv(ml1, 'train')
    env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                 env=train_env,
                                 wrapper=lambda env, _: normalize(env))
    env = env_sampler.sample(num_train_tasks)
    test_env = MetaWorldSetTaskEnv(ml1, 'test')
    test_env_sampler = SetTaskSampler(MetaWorldSetTaskEnv,
                                      env=test_env,
                                      wrapper=lambda env, _: normalize(env),
                                      )


    trainer = Trainer(ctxt)

    # instantiate networks
    augmented_env = PEARL.augment_env_spec(env[0](), latent_size)
    qf1 = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])
    qf2 = ContinuousMLPQFunction(env_spec=augmented_env,
                                hidden_sizes=[net_size, net_size, net_size])

    vf_env = PEARL.get_env_spec(env[0](), latent_size, 'vf')
    vf = ContinuousMLPQFunction(env_spec=vf_env,
                                hidden_sizes=[net_size, net_size, net_size])

    inner_policy = TanhGaussianMLPPolicy(
        env_spec=augmented_env, hidden_sizes=[net_size, net_size, net_size])

    sampler = LocalSampler(agents=None,
                           envs=env[0](),
                           max_episode_length=env[0]().spec.max_episode_length,
                           n_workers=1,
                           worker_class=PEARLWorker)

    pearl = PEARL(
        env=env,
        policy_class=ContextConditionedPolicy,
        encoder_class=MLPEncoder,
        inner_policy=inner_policy,
        qf1=qf1,
        qf2=qf2,
        vf=vf,
        sampler=sampler,
        num_train_tasks=num_train_tasks,
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
        reward_scale=reward_scale,
        num_test_tasks=5,
        replay_buffer_size=int(5e6)
    )

    set_gpu_mode(use_gpu, gpu_id=0)
    if use_gpu:
        pearl.to()

    trainer.setup(algo=pearl, env=env[0]())

    trainer.train(n_epochs=num_epochs, batch_size=batch_size)


pearl_metaworld_ml1()