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
from garage.np.exploration_strategies import OUStrategy
from garage.replay_buffer import HerReplayBuffer
from garage.tf.algos import DDPG
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        env = TfEnv(gym.make('FetchReach-v1'))

        action_noise = OUStrategy(env.spec, sigma=0.2)

        policy = ContinuousMLPPolicy(
            env_spec=env.spec,
            name='Policy',
            hidden_sizes=[256, 256, 256],
            hidden_nonlinearity=tf.nn.relu,
            output_nonlinearity=tf.nn.tanh,
            input_include_goal=True,
        )

        qf = ContinuousMLPQFunction(
            env_spec=env.spec,
            name='QFunction',
            hidden_sizes=[256, 256, 256],
            hidden_nonlinearity=tf.nn.relu,
            input_include_goal=True,
        )

        replay_buffer = HerReplayBuffer(env_spec=env.spec,
                                        size_in_transitions=int(1e6),
                                        time_horizon=100,
                                        replay_k=0.4,
                                        reward_fun=env.compute_reward)

        ddpg = DDPG(
            env_spec=env.spec,
            policy=policy,
            policy_lr=1e-3,
            qf_lr=1e-3,
            qf=qf,
            replay_buffer=replay_buffer,
            target_update_tau=0.05,
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

        runner.setup(algo=ddpg, env=env)

        runner.train(n_epochs=50, batch_size=100, n_epoch_cycles=20)


run_experiment(run_task, snapshot_mode='last', seed=1)
