"""
Unit test for DeterministicMLPPolicyWithModel.

This test consists of four different DeterministicMLPPolicy: P1, P2, P3
and P4. P1 and P2 are from DeterministicMLPPolicy, which does not use
garage.tf.models.MLPModel while P3 and P4 do use.

This test ensures the outputs from all the policies are the same,
for the transition from using DeterministicMLPPolicy to
DeterministicMLPPolicyWithModel.

It covers get_action, get_actions, get_action_sym.
"""
from unittest import mock

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.policies import DeterministicMLPPolicy
from garage.tf.policies import DeterministicMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestDeterministicMLPPolicyWithModelTransit(TfGraphTestCase):
    @mock.patch('tensorflow.random.normal')
    def setUp(self, mock_rand):
        mock_rand.return_value = 0.5
        super().setUp()
        self.box_env = TfEnv(DummyBoxEnv())
        self.policy1 = DeterministicMLPPolicy(
            env_spec=self.box_env, hidden_sizes=(32, 32), name='P1')
        self.policy2 = DeterministicMLPPolicy(
            env_spec=self.box_env, hidden_sizes=(64, 64), name='P2')
        self.policy3 = DeterministicMLPPolicyWithModel(
            env_spec=self.box_env, hidden_sizes=(32, 32), name='P3')
        self.policy4 = DeterministicMLPPolicyWithModel(
            env_spec=self.box_env, hidden_sizes=(64, 64), name='P4')

        self.sess.run(tf.global_variables_initializer())
        for a, b in zip(self.policy3.get_params(), self.policy1.get_params()):
            self.sess.run(b.assign(a))
        for a, b in zip(self.policy4.get_params(), self.policy2.get_params()):
            self.sess.run(b.assign(a))

        self.obs = [self.box_env.reset()]

        assert self.policy1.vectorized == self.policy2.vectorized
        assert self.policy3.vectorized == self.policy4.vectorized

    @mock.patch('numpy.random.normal')
    def test_get_action(self, mock_rand):
        mock_rand.return_value = 0.5
        action1, _ = self.policy1.get_action(self.obs)
        action2, _ = self.policy2.get_action(self.obs)
        action3, _ = self.policy3.get_action(self.obs)
        action4, _ = self.policy4.get_action(self.obs)

        assert np.array_equal(action1, action3)
        assert np.array_equal(action2, action4)

        actions1, _ = self.policy1.get_actions([self.obs, self.obs])
        actions2, _ = self.policy2.get_actions([self.obs, self.obs])
        actions3, _ = self.policy3.get_actions([self.obs, self.obs])
        actions4, _ = self.policy4.get_actions([self.obs, self.obs])

        assert np.array_equal(actions1, actions3)
        assert np.array_equal(actions2, actions4)

    def test_get_action_sym(self):
        obs_dim = self.box_env.spec.observation_space.flat_dim
        state_input = tf.placeholder(tf.float32, shape=(None, obs_dim))

        action_sym1 = self.policy1.get_action_sym(
            state_input, name='action_sym')
        action_sym2 = self.policy2.get_action_sym(
            state_input, name='action_sym')
        action_sym3 = self.policy3.get_action_sym(
            state_input, name='action_sym')
        action_sym4 = self.policy4.get_action_sym(
            state_input, name='action_sym')

        action1 = self.sess.run(action_sym1, feed_dict={state_input: self.obs})
        action2 = self.sess.run(action_sym2, feed_dict={state_input: self.obs})
        action3 = self.sess.run(action_sym3, feed_dict={state_input: self.obs})
        action4 = self.sess.run(action_sym4, feed_dict={state_input: self.obs})

        assert np.array_equal(action1, action3)
        assert np.array_equal(action2, action4)
