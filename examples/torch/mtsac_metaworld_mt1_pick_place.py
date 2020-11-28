#!/usr/bin/env python3
"""MTSAC implementation based on Metaworld. Benchmarked on MT1.

This experiment shows how MTSAC adapts to 50 environents of the same type
but each environment has a goal variation.

https://arxiv.org/pdf/1910.10897.pdf
"""
import click
import metaworld
import numpy as np
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic, MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer


@click.command()
@click.option('--seed', 'seed', type=int, default=1)
@click.option('--timesteps', default=10000000)
@click.option('--gpu', '_gpu', type=int, default=None)
@wrap_experiment(snapshot_mode='none')
def mtsac_metaworld_mt1_pick_place(ctxt=None, *, seed, timesteps, _gpu):
    """Train MTSAC with the MT1 pick-place-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).
        timesteps (int): Number of timesteps to run.

    """
    deterministic.set_seed(seed)
    mt1 = metaworld.MT1('pick-place-v1')
    mt1_test = metaworld.MT1('pick-place-v1')
    train_task_sampler = MetaWorldTaskSampler(mt1, 'train',
                                              lambda env, _: normalize(env))
    test_task_sampler = MetaWorldTaskSampler(mt1_test, 'train',
                                             lambda env, _: normalize(env))
    n_tasks = 50
    train_envs = train_task_sampler.sample(n_tasks)
    env = train_envs[0]()
    test_envs = [env_up() for env_up in test_task_sampler.sample(n_tasks)]

    trainer = Trainer(ctxt)

    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[400, 400, 400],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )

    sampler = LocalSampler(agents=policy,
                           envs=train_envs,
                           max_episode_length=env.spec.max_episode_length,
                           n_workers=n_tasks,
                           worker_class=FragmentWorker)

    batch_size = int(env.spec.max_episode_length * n_tasks)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  sampler=sampler,
                  gradient_steps_per_itr=150,
                  eval_env=test_envs,
                  env_spec=env.spec,
                  num_tasks=1,
                  steps_per_epoch=epoch_cycles,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1500,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=1280)
    if _gpu is not None:
        set_gpu_mode(True, _gpu)
    mtsac.to()
    trainer.setup(algo=mtsac, env=train_envs)
    trainer.train(n_epochs=epochs, batch_size=batch_size)


# pylint: disable=missing-kwoa
mtsac_metaworld_mt1_pick_place()
