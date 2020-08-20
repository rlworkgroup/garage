#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import click
import gym
from metaworld.envs.mujoco.env_dict import (ALL_V1_ENVIRONMENTS,
                                            ALL_V2_ENVIRONMENTS)
from metaworld.policies import *
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GymEnv, normalize
from garage.envs.wrappers import MWScriptedReward
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


@click.command()
@click.option('--env-name', type=str)
@click.option('--seed', type=int, default=np.random.randint(0, 1000))
@click.option('--gpu', type=int, default=0)
@wrap_experiment(snapshot_mode='gap', snapshot_gap=50, name_parameters='all')
def sac_metaworld_new_reward_function(ctxt=None, env_name=None, seed=1, gpu=None, reward_scale=10):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.

    """
    not_in_mw = 'the env_name specified is not a metaworld environment'
    assert env_name in ALL_V2_ENVIRONMENTS or env_name in ALL_V1_ENVIRONMENTS, not_in_mw
    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)

    if env_name in ALL_V2_ENVIRONMENTS:
        env_cls = ALL_V2_ENVIRONMENTS[env_name]
    else:
        env_cls = ALL_V1_ENVIRONMENTS[env_name]

    env = env_cls()
    env._partially_observable = False
    env._freeze_rand_vec = False
    env._set_task_called = True
    env.reset()
    env._freeze_rand_vec = True
    env.max_path_length = 500
    max_path_length = env.max_path_length

    env = GymEnv(env)
    policy = TanhGaussianMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))
    batch_size = 50

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=batch_size,
              max_episode_length=max_path_length,
              max_episode_length_eval=max_path_length,
              replay_buffer=replay_buffer,
              min_buffer_size=1e4,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=256,
              reward_scale=float(reward_scale),
              steps_per_epoch=800,
              num_evaluation_episodes=10)

    if gpu is not None:
        set_gpu_mode(True, gpu)
    sac.to()
    runner.setup(algo=sac, env=env, sampler_cls=LocalSampler)
    runner.train(n_epochs=250, batch_size=batch_size)


sac_metaworld_new_reward_function()
