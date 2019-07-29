#!/usr/bin/env python3
"""
An example to train a task with DQN algorithm.

Here it creates a gym environment CartPole, and trains a DQN with 50k steps.
"""
import gym

from garage.experiment import run_experiment
from garage.np.exploration_strategies import EpsilonGreedyStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction


def run_task(snapshot_config, *_):
    """Run task."""
    with LocalTFRunner(snapshot_config=snapshot_config) as runner:
        n_epochs = 10
        n_epoch_cycles = 10
        sampler_batch_size = 500
        num_timesteps = n_epochs * n_epoch_cycles * sampler_batch_size
        env = TfEnv(gym.make('CartPole-v0'))
        replay_buffer = SimpleReplayBuffer(env_spec=env.spec,
                                           size_in_transitions=int(1e4),
                                           time_horizon=1)
        qf = DiscreteMLPQFunction(env_spec=env.spec, hidden_sizes=(64, 64))
        policy = DiscreteQfDerivedPolicy(env_spec=env.spec, qf=qf)
        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env.spec,
            total_timesteps=num_timesteps,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)
        algo = DQN(env_spec=env.spec,
                   policy=policy,
                   qf=qf,
                   exploration_strategy=epilson_greedy_strategy,
                   replay_buffer=replay_buffer,
                   qf_lr=1e-4,
                   discount=1.0,
                   min_buffer_size=int(1e3),
                   double_q=True,
                   n_train_steps=500,
                   n_epoch_cycles=n_epoch_cycles,
                   target_network_update_freq=1,
                   buffer_batch_size=32)

        runner.setup(algo, env)
        runner.train(n_epochs=n_epochs,
                     n_epoch_cycles=n_epoch_cycles,
                     batch_size=sampler_batch_size)


run_experiment(run_task, snapshot_mode='last', seed=1)
