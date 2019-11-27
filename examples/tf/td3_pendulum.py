#!/usr/bin/env python3
"""This is an example to train a task with TD3 algorithm.

Here, we create a gym environment InvertedDoublePendulum
and use a TD3 with 1M steps.

Results:
    AverageReturn: 250
    RiseTime: epoch 499
"""
import gym
import tensorflow as tf

from garage.experiment import run_experiment
from garage.np.exploration_strategies.gaussian_strategy import GaussianStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import TD3
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import ContinuousMLPPolicy
from garage.tf.q_functions import ContinuousMLPQFunction


def run_task(snapshot_config, *_):
    """Wrap TD3 training task in the run_task function.

    Args:
        snapshot_config (garage.experiment.SnapshotConfig): Configuration
            values for snapshotting.
        *_ (object): Hyperparameters (unused).

    """
    with LocalTFRunner(snapshot_config) as runner:
        env = TfEnv(gym.make('InvertedDoublePendulum-v2'))

        action_noise = GaussianStrategy(env.spec, max_sigma=0.1, min_sigma=0.1)

        policy = ContinuousMLPPolicy(env_spec=env.spec,
                                     hidden_sizes=[400, 300],
                                     hidden_nonlinearity=tf.nn.relu,
                                     output_nonlinearity=tf.nn.tanh)

        qf = ContinuousMLPQFunction(name='ContinuousMLPQFunction',
                                    env_spec=env.spec,
                                    hidden_sizes=[400, 300],
                                    action_merge_layer=0,
                                    hidden_nonlinearity=tf.nn.relu)

        qf2 = ContinuousMLPQFunction(name='ContinuousMLPQFunction2',
                                     env_spec=env.spec,
                                     hidden_sizes=[400, 300],
                                     action_merge_layer=0,
                                     hidden_nonlinearity=tf.nn.relu)

        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e6),
                                           time_horizon=250)

        td3 = TD3(env_spec=env.spec,
                  policy=policy,
                  policy_lr=1e-4,
                  qf_lr=1e-3,
                  qf=qf,
                  qf2=qf2,
                  replay_buffer=replay_buffer,
                  target_update_tau=1e-2,
                  n_epoch_cycles=20,
                  n_train_steps=1,
                  smooth_return=False,
                  discount=0.99,
                  buffer_batch_size=100,
                  min_buffer_size=1e4,
                  exploration_strategy=action_noise,
                  policy_optimizer=tf.train.AdamOptimizer,
                  qf_optimizer=tf.train.AdamOptimizer)

        runner.setup(td3, env)
        runner.train(n_epochs=500, n_epoch_cycles=20, batch_size=250)


run_experiment(
    run_task,
    snapshot_mode='last',
    seed=1,
)
