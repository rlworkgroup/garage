#!/usr/bin/env python3
"""
This is an example to train a task with DDPG algorithm.

Here it creates a gym environment InvertedDoublePendulum. And uses a DDPG with
1M steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 499
"""
import tensorflow as tf

from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.exploration_strategies import OUStrategy
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction
from garage.runners.local_tf_runner import LocalRunner
from garage.tf.samplers.off_policy_vectorized_sampler import OffPolicyVectorizedSampler

with LocalRunner() as runner:
    env = TfEnv(env_name="Pendulum-v0")

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

    n_epochs = 500
    n_epoch_cycles = 20
    max_path_length = 100

    ddpg = DDPG(
        env,
        policy=policy,
        policy_lr=1e-4,
        qf_lr=1e-3,
        qf=qf,
        replay_buffer=replay_buffer,
        target_update_tau=1e-2,
        n_epoch_cycles=n_epoch_cycles,
        max_path_length=100,
        n_train_steps=50,
        discount=0.9,
        min_buffer_size=int(1e4),
        exploration_strategy=action_noise,
        policy_optimizer=tf.train.AdamOptimizer,
        qf_optimizer=tf.train.AdamOptimizer)

    runner.setup(algo=ddpg, env=env, sampler_cls=OffPolicyVectorizedSampler)

    runner.train(n_epochs=n_epochs, n_epoch_cycles=n_epoch_cycles)
