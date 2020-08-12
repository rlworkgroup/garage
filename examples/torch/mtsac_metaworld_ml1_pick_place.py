#!/usr/bin/env python3
"""MTSAC implementation based on Metaworld. Benchmarked on ML1.

This experiment shows how MTSAC adapts to 50 environents of the same type
but each environment has a goal variation.

https://arxiv.org/pdf/1910.10897.pdf
"""
import pickle

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
def mtsac_metaworld_ml1_pick_place(ctxt=None, seed=1, _gpu=None):
    """Train MTSAC with the ML1 pick-place-v1 environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        _gpu (int): The ID of the gpu to be used (used on multi-gpu machines).

    """
    deterministic.set_seed(seed)
    runner = LocalRunner(ctxt)
    train_envs = []
    test_envs = []
    env_names = []
    for i in range(50):
        train_env = normalize(
            GymEnv(mwb.ML1.get_train_tasks('pick-place-v1'),
                   normalize_reward=True))
        test_env = pickle.loads(pickle.dumps(train_env))
        env_names.append('pick_place_{}'.format(i))
        train_envs.append(train_env)
        test_envs.append(test_env)
    ml1_train_envs = MultiEnvWrapper(train_envs,
                                     sample_strategy=round_robin_strategy,
                                     env_names=env_names)
    ml1_test_envs = MultiEnvWrapper(test_envs,
                                    sample_strategy=round_robin_strategy,
                                    env_names=env_names)
    policy = TanhGaussianMLPPolicy(
        env_spec=ml1_train_envs.spec,
        hidden_sizes=[400, 400, 400],
        hidden_nonlinearity=nn.ReLU,
        output_nonlinearity=None,
        min_std=np.exp(-20.),
        max_std=np.exp(2.),
    )

    qf1 = ContinuousMLPQFunction(env_spec=ml1_train_envs.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=ml1_train_envs.spec,
                                 hidden_sizes=[400, 400, 400],
                                 hidden_nonlinearity=F.relu)
    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6), )

    timesteps = 10000000
    batch_size = int(150 * ml1_train_envs.num_tasks)
    num_evaluation_points = 500
    epochs = timesteps // batch_size
    epoch_cycles = epochs // num_evaluation_points
    epochs = epochs // epoch_cycles
    mtsac = MTSAC(policy=policy,
                  qf1=qf1,
                  qf2=qf2,
                  gradient_steps_per_itr=150,
                  max_episode_length=150,
                  eval_env=ml1_test_envs,
                  env_spec=ml1_train_envs.spec,
                  num_tasks=50,
                  steps_per_epoch=epoch_cycles,
                  replay_buffer=replay_buffer,
                  min_buffer_size=1500,
                  target_update_tau=5e-3,
                  discount=0.99,
                  buffer_batch_size=1280)
    if _gpu is not None:
        set_gpu_mode(True, _gpu)
    mtsac.to()
    runner.setup(algo=mtsac, env=ml1_train_envs, sampler_cls=LocalSampler)
    runner.train(n_epochs=epochs, batch_size=batch_size)


mtsac_metaworld_ml1_pick_place()
