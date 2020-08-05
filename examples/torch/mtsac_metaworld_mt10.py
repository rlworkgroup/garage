#!/usr/bin/env python3
"""MTSAC implementation based on Metaworld. Benchmarked on MT10.

https://arxiv.org/pdf/1910.10897.pdf
"""
import click
import metaworld.benchmarks as mwb
import numpy as np
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv, MultiEnvWrapper, normalize
from garage.envs.multi_env_wrapper import round_robin_strategy
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import MTSAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


@click.command()
@click.option('--seed', 'seed', type=int, default=1)
@click.option('--gpu', '_gpu', type=int, default=None)
@wrap_experiment(snapshot_mode='none')
def mtsac_metaworld_mt10(ctxt=None, seed=1, _gpu=None):
    """Train MTSAC with MT10 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).

    """
    deterministic.set_seed(seed)
    runner = LocalRunner(ctxt)
    task_names = mwb.MT10.get_train_tasks().all_task_names
    train_envs = []
    test_envs = []
    for task_name in task_names:
        train_env = normalize(GymEnv(mwb.MT10.from_task(task_name)),
                              normalize_reward=True)
        test_env = normalize(GymEnv(mwb.MT10.from_task(task_name)))
        train_envs.append(train_env)
        test_envs.append(test_env)
    mt10_train_envs = MultiEnvWrapper(train_envs,
                                      sample_strategy=round_robin_strategy,
                                      mode='vanilla')
    mt10_test_envs = MultiEnvWrapper(test_envs,
                                     sample_strategy=round_robin_strategy,
                                     mode='vanilla')
    policy = TanhGaussianMLPPolicy(
        env_spec=mt10_train_envs.spec,
        hidden_sizes=[400, 400, 400],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=mt10_train_envs.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=mt10_train_envs.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )

    timesteps = 20000000
    batch_size = int(150 * mt10_train_envs.num_tasks)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=150,
                  max_episode_length=150,
                  eval_env=mt10_test_envs,
                  env_spec=mt10_train_envs.spec,
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
    runner.setup(algo=mtsac, env=mt10_train_envs, sampler_cls=LocalSampler)
    runner.train(n_epochs=epochs, batch_size=batch_size)


mtsac_metaworld_mt10()
