#!/usr/bin/env python3
"""MTSAC implementation based on Metaworld. Benchmarked on MT10.

https://arxiv.org/pdf/1910.10897.pdf
"""
import click
import metaworld
import numpy as np
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import normalize
from garage.experiment import deterministic
from garage.experiment.task_sampler import MetaWorldTaskSampler
from garage.replay_buffer import PathBuffer
from garage.sampler import FragmentWorker, LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
from garage.trainer import Trainer


@click.command()
@click.option('--seed', 'seed', type=int, default=1)
@click.option('--gpu', '_gpu', type=int, default=None)
@click.option('--n_tasks', default=10)
@click.option('--timesteps', default=20000000)
@wrap_experiment(snapshot_mode='none')
def mtsac_metaworld_mt10(ctxt=None, *, seed, _gpu, n_tasks, timesteps):
    """Train MTSAC with MT10 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by Trainer to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).
        n_tasks (int): Number of tasks to use. Should be a multiple of 10.
        timesteps (int): Number of timesteps to run.

    """
    deterministic.set_seed(seed)
    trainer = Trainer(ctxt)
    mt10 = metaworld.MT10()  # pylint: disable=no-member
    mt10_test = metaworld.MT10()  # pylint: disable=no-member

    # pylint: disable=missing-return-doc, missing-return-type-doc
    def wrap(env, _):
        return normalize(env, normalize_reward=True)

    train_task_sampler = MetaWorldTaskSampler(mt10,
                                              'train',
                                              wrap,
                                              add_env_onehot=True)
    test_task_sampler = MetaWorldTaskSampler(mt10_test,
                                             'train',
                                             add_env_onehot=True)
    assert n_tasks % 10 == 0
    assert n_tasks <= 500
    mt10_train_envs = train_task_sampler.sample(n_tasks)
    env = mt10_train_envs[0]()
    mt10_test_envs = [env_up() for env_up in test_task_sampler.sample(n_tasks)]

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
    meta_batch_size = 10

    sampler = LocalSampler(
        agents=policy,
        envs=mt10_train_envs,
        max_episode_length=env.spec.max_episode_length,
        # 1 sampler worker for each environment
        n_workers=meta_batch_size,
        worker_class=FragmentWorker,
        # increasing n_envs increases the vectorization of a sampler worker
        # which improves runtime performance, but you will need to adjust this
        # depending on your memory constraints. For reference, each worker by
        # default uses n_envs=8. Each environment is approximately ~50mb large
        # so creating 50 envs with 8 copies comes out to 20gb of memory. Many
        # users want to be able to run multiple seeds on 1 machine, so I have
        # reduced this to n_envs = 2 for 2 copies in the meantime.
        worker_args=dict(n_envs=2))

    batch_size = int(env.spec.max_episode_length * meta_batch_size)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  sampler=sampler,
                  gradient_steps_per_itr=env.spec.max_episode_length,
                  eval_env=mt10_test_envs,
                  env_spec=env.spec,
                  num_tasks=10,
                  steps_per_epoch=epoch_cycles,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1500,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=1280)
    if _gpu is not None:
        set_gpu_mode(True, _gpu)
    mtsac.to()
    trainer.setup(algo=mtsac, env=mt10_train_envs)
    trainer.train(n_epochs=epochs, batch_size=batch_size)


# pylint: disable=missing-kwoa
mtsac_metaworld_mt10()
