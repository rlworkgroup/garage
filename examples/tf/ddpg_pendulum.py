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
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('InvertedDoublePendulum-v2'))

        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=[64, 64],
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(env_spec=env.spec,
                                    hidden_sizes=[64, 64],
                                    hidden_nonlinearity=tf.nn.relu)

        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e6),
                                           time_horizon=100)

        ddpg = DDPG(env_spec=env.spec,
                    policy=policy,
                    policy_lr=1e-4,
                    qf_lr=1e-3,
                    qf=qf,
                    replay_buffer=replay_buffer,
                    target_update_tau=1e-2,
                    n_train_steps=50,
                    discount=0.9,
                    min_buffer_size=int(1e4),
                    exploration_strategy=action_noise,
                    policy_optimizer=tf.train.AdamOptimizer,
                    qf_optimizer=tf.train.AdamOptimizer)

        runner.setup(algo=ddpg, env=env)

        runner.train(n_epochs=500, n_epoch_cycles=20, batch_size=100)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
