#!/usr/bin/env python3
"""
This is an example to train a task with DDPG + HER algorithm.

Here it creates a gym environment FetchReach.

Results (may vary by seed):
    AverageSuccessRate: 0.9
    RiseTime: epoch 8
"""
import gym
import tensorflow as tf

from garage.experiment import run_experiment
from garage.replay_buffer import HerReplayBuffer
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
    env = TfEnv(gym.make('FetchReach-v1'))

    action_noise = OUStrategy(env.spec, sigma=0.2)

    policy = ContinuousMLPPolicy(
        env_spec=env.spec,
        name="Policy",
        hidden_sizes=[256, 256, 256],
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=tf.nn.tanh,
        input_include_goal=True,
    )

    qf = ContinuousMLPQFunction(
        env_spec=env.spec,
        name="QFunction",
        hidden_sizes=[256, 256, 256],
        hidden_nonlinearity=tf.nn.relu,
        input_include_goal=True,
    )

    replay_buffer = HerReplayBuffer(
        env_spec=env.spec,
        size_in_transitions=int(1e6),
        time_horizon=100,
        replay_k=0.4,
        reward_fun=env.compute_reward)

    ddpg = DDPG(
        env,
        policy=policy,
        policy_lr=1e-3,
        qf_lr=1e-3,
        qf=qf,
        replay_buffer=replay_buffer,
        plot=False,
        target_update_tau=0.05,
        n_epochs=50,
        n_epoch_cycles=20,
        max_path_length=100,
        n_train_steps=40,
        discount=0.9,
        exploration_strategy=action_noise,
        policy_optimizer=tf.train.AdamOptimizer,
        qf_optimizer=tf.train.AdamOptimizer,
        buffer_batch_size=256,
        input_include_goal=True,
    )

    ddpg.train()


run_experiment(
    run_task,
    n_parallel=1,
    snapshot_mode="last",
    seed=1,
    plot=False,
)
