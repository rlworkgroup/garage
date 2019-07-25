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
from garage.experiment import LocalRunner, run_experiment
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.envs import TfEnv
from garage.torch.algos import DDPG
from garage.torch.modules import MLPModule
from garage.torch.policies import DeterministicPolicy
from garage.torch.q_functions import ContinuousNNQFunction


def run_task(*_):
    """Set up environment and algorithm and run the task."""
    with LocalRunner() as runner:
        env = TfEnv(normalize(gym.make('InvertedDoublePendulum-v2')))

        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy_module = MLPModule(
            input_dim=env.spec.observation_space.flat_dim,
            output_dim=env.spec.action_space.flat_dim,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=F.relu,
            output_nonlinearity=torch.tanh)
        policy = DeterministicPolicy(env.spec, policy_module)

        input_dim = (env.spec.observation_space.flat_dim +
                     env.spec.action_space.flat_dim)

        qf_module = MLPModule(
            input_dim=input_dim,
            output_dim=1,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=F.relu)
        qf = ContinuousNNQFunction(env.spec, qf_module)

        target_policy_module = MLPModule(
            input_dim=env.spec.observation_space.flat_dim,
            output_dim=env.spec.action_space.flat_dim,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=F.relu,
            output_nonlinearity=torch.tanh)
        target_policy = DeterministicPolicy(env.spec, target_policy_module)

        target_qf_module = MLPModule(
            input_dim=input_dim,
            output_dim=1,
            hidden_sizes=[64, 64],
            hidden_nonlinearity=F.relu)
        target_qf = ContinuousNNQFunction(env.spec, target_qf_module)

        replay_buffer = SimpleReplayBuffer(
            env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

        ddpg = DDPG(
            env_spec=env.spec,
            policy=policy,
            target_policy=target_policy,
            qf=qf,
            target_qf=target_qf,
            replay_buffer=replay_buffer,
            n_train_steps=50,
            min_buffer_size=int(1e4),
            exploration_strategy=action_noise,
            target_update_tau=1e-2,
            policy_lr=1e-4,
            qf_lr=1e-3,
            discount=0.9,
            optimizer=torch.optim.Adam)

        runner.setup(algo=ddpg, env=env)

        runner.train(n_epochs=500, n_epoch_cycles=20, batch_size=100)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
