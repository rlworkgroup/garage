#!/usr/bin/env python3
"""
This is an example to train a task with DDPG algorithm written in PyTorch.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

"""
import gym
import torch
from torch.nn import functional as F  # NOQA

from garage.envs import normalize
from garage.envs.base import GarageEnv
from garage.experiment import LocalRunner, run_experiment
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.torch.algos import SAC
from garage.torch.policies import GaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction


def run_task(snapshot_config, *_):
    """Set up environment and algorithm and run the task."""
    runner = LocalRunner(snapshot_config)
    env = GarageEnv(normalize(gym.make('InvertedDoublePendulum-v2')))

    action_noise = OUStrategy(env.spec, sigma=0.2)

    policy = GaussianMLPPolicy(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=F.relu,
                                    output_nonlinearity=torch.tanh)

    qf1 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[64, 64],
                                hidden_nonlinearity=F.relu)

    qf2 = ContinuousMLPQFunction(env_spec=env.spec,
                                hidden_sizes=[64, 64],
                                hidden_nonlinearity=F.relu)

    replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                       size_in_transitions=int(1e6),
                                       time_horizon=100)

    SAC = SAC(env_spec=env.spec,
                policy=policy,
                qf1=qf1,
                qf2=qf2,
                alpha=0.1,
                replay_buffer=replay_buffer,
                target_update_tau=1e-2,
                discount=0.9)

    runner.setup(algo=SAC, env=env)

    runner.train(n_epochs=1, n_epoch_cycles=20, batch_size=100)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
