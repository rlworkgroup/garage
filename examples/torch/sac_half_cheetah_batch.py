#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import gym
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # NOQA

from garage import wrap_experiment
from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import SimpleReplayBuffer
from garage.sampler import LocalSampler
from garage.torch.algos import SAC
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction
import garage.torch.utils as tu


@wrap_experiment(snapshot_mode='none')
def torch_sac_half_cheetah(ctxt=None, seed=1):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.

    """
    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    env = GarageEnv(normalize(gym.make('HalfCheetah-v2')))

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

    replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                       size_in_transitions=int(1e6),
                                       time_horizon=1)

    sac = SAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=1000,
              max_path_length=500,
              use_automatic_entropy_tuning=True,
              replay_buffer=replay_buffer,
              min_buffer_size=1e4,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=256,
              reward_scale=1.,
              steps_per_epoch=2)

    if torch.cuda.is_available():
        tu.set_gpu_mode(True)
    else:
        tu.set_gpu_mode(False)
    sac.to()
    runner.setup(algo=sac, env=env, sampler_cls=LocalSampler)
    runner.train(n_epochs=500, batch_size=1000)


s = np.random.randint(0, 1000)
torch_sac_half_cheetah(seed=s)
