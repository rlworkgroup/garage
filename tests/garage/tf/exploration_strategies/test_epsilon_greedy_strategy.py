"""
This script creates a unittest that tests Gaussian policies in
garage.tf.policies.
"""
from unittest import mock

import numpy as np
import tensorflow as tf

from garage.tf.core.mlp import mlp
from garage.tf.envs import TfEnv
from garage.tf.exploration_strategies import EpsilonGreedyStrategy
from garage.tf.policies import DiscreteQfDerivedPolicy
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestEpsilonGreedyStrategy(TfGraphTestCase):
    def test_epsilon_greedy_strategy(self):
        env = TfEnv(DummyDiscreteEnv())
        # mock a q_function
        obs_ph = tf.placeholder(
            tf.float32, shape=(None, ) + env.observation_space.shape)
        qf_function = mock.Mock()
        qf_function.q_val = mlp(
            input_var=obs_ph,
            output_dim=env.action_space.flat_dim,
            hidden_sizes=(32, 32),
            hidden_nonlinearity=tf.nn.relu,
            name="mlp")
        qf_function.obs_ph = obs_ph

        policy = DiscreteQfDerivedPolicy(env_spec=env, qf=qf_function)

        epilson_greedy_strategy = EpsilonGreedyStrategy(
            env_spec=env,
            total_step=100,
            max_epsilon=1.0,
            min_epsilon=0.02,
            decay_ratio=0.1)

        self.sess.run(tf.global_variables_initializer())

        obs, _, _, _ = env.step(1)

        action = epilson_greedy_strategy.get_action(0, obs, policy, self.sess)
        assert action in np.arange(env.action_space.n)
        actions = epilson_greedy_strategy.get_actions(0, [obs], policy,
                                                      self.sess)
        assert actions in np.arange(env.action_space.n)
