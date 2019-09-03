"""
This script creates a test that fails when garage.tf.algos.DQN performance is
too low.
"""
import pickle

import gym
import numpy as np
import pytest
import tensorflow as tf

from garage.np.exploration_strategies import EpsilonGreedyStrategy
from garage.replay_buffer import SimpleReplayBuffer
from garage.tf.algos import DQN
from garage.tf.envs import TfEnv
from garage.tf.experiment import LocalTFRunner
from garage.tf.policies import DiscreteQfDerivedPolicy
from garage.tf.q_functions import DiscreteMLPQFunction
from tests.fixtures import snapshot_config, TfGraphTestCase


class TestDQN(TfGraphTestCase):

    @pytest.mark.large
    def test_dqn_cartpole(self):
        """Test DQN with CartPole environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
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
                       double_q=False,
                       n_train_steps=500,
                       n_epoch_cycles=n_epoch_cycles,
                       target_network_update_freq=1,
                       buffer_batch_size=32)

            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=n_epochs,
                                        n_epoch_cycles=n_epoch_cycles,
                                        batch_size=sampler_batch_size)
            assert last_avg_ret > 15

            env.close()

    @pytest.mark.large
    def test_dqn_cartpole_double_q(self):
        """Test DQN with CartPole environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
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
            last_avg_ret = runner.train(n_epochs=n_epochs,
                                        n_epoch_cycles=n_epoch_cycles,
                                        batch_size=sampler_batch_size)
            assert last_avg_ret > 15

            env.close()

    @pytest.mark.large
    def test_dqn_cartpole_grad_clip(self):
        """Test DQN with CartPole environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
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
                       double_q=False,
                       n_train_steps=500,
                       grad_norm_clipping=5.0,
                       n_epoch_cycles=n_epoch_cycles,
                       target_network_update_freq=1,
                       buffer_batch_size=32)

            runner.setup(algo, env)
            last_avg_ret = runner.train(n_epochs=n_epochs,
                                        n_epoch_cycles=n_epoch_cycles,
                                        batch_size=sampler_batch_size)
            assert last_avg_ret > 15

            env.close()

    def test_dqn_cartpole_pickle(self):
        """Test DQN with CartPole environment."""
        with LocalTFRunner(snapshot_config, sess=self.sess) as runner:
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
                       double_q=False,
                       n_train_steps=500,
                       grad_norm_clipping=5.0,
                       n_epoch_cycles=n_epoch_cycles,
                       target_network_update_freq=1,
                       buffer_batch_size=32)
            runner.setup(algo, env)
            with tf.compat.v1.variable_scope(
                    'DiscreteMLPQFunction/MLPModel/mlp/hidden_0', reuse=True):
                bias = tf.compat.v1.get_variable('bias')
                # assign it to all one
                old_bias = tf.ones_like(bias).eval()
                bias.load(old_bias)
                h = pickle.dumps(algo)

            with tf.compat.v1.Session(graph=tf.Graph()):
                pickle.loads(h)
                with tf.compat.v1.variable_scope(
                        'DiscreteMLPQFunction/MLPModel/mlp/hidden_0',
                        reuse=True):
                    new_bias = tf.compat.v1.get_variable('bias')
                    new_bias = new_bias.eval()
                    assert np.array_equal(old_bias, new_bias)

            env.close()
