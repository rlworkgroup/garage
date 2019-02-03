#!/usr/bin/env python3
"""
This is an example to train a task with DDPG algorithm.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 499
"""
import gym
import tensorflow as tf

from garage.experiment import run_experiment
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def run_task(*_):
    """
    Wrap DDPG training task in the run_task function.

    :param _:
    :return:
    """
    env = TfEnv(gym.make('InvertedDoublePendulum-v2'))

    action_noise = OUStrategy(env.spec, sigma=0.2)

    policy = ContinuousMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[64, 64],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh)

    qf = ContinuousMLPQFunction(
        env_spec=env.spec,
        hidden_sizes=[64, 64],
        hidden_nonlinearity=tf.nn.relu)

    replay_buffer = SimpleReplayBuffer(
        env_spec=env.spec, size_in_transitions=int(1e6), time_horizon=100)

    ddpg = DDPG(
        env,
        policy=policy,
        policy_lr=1e-4,
        qf_lr=1e-3,
        qf=qf,
        replay_buffer=replay_buffer,
        plot=False,
        target_update_tau=1e-2,
        n_epochs=500,
        n_epoch_cycles=20,
        max_path_length=100,
        n_train_steps=50,
        discount=0.9,
        min_buffer_size=int(1e4),
        exploration_strategy=action_noise,
        policy_optimizer=tf.train.AdamOptimizer,
        qf_optimizer=tf.train.AdamOptimizer)

    ddpg.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
