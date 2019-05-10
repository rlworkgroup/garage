"""
Unit test for Categorical LSTM Policy with Model.

This test consists of four different CategoricalLSTMPolicy: P1, P2, P3
and P4. P1 and P2 are from CategoricalLSTMPolicy, which does not use
garage.tf.models.LSTMModel while P3 and P4 do use.

This test ensures the outputs from all the policies are the same,
for the transition from using CategoricalLSTMPolicy to
CategoricalLSTMPolicyWithModel.

It covers get_action, get_actions, dist_info_sym, kl_sym,
log_likelihood_sym, entropy_sym and likelihood_ratio_sym.
"""
from unittest import mock

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.misc import tensor_utils
from garage.tf.policies import CategoricalLSTMPolicy
from garage.tf.policies import CategoricalLSTMPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyDiscreteEnv


class TestCategoricalLSTMPolicyWithModelTransit(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        env = TfEnv(DummyDiscreteEnv(obs_dim=(1, ), action_dim=1))
        self.default_initializer = tf.constant_initializer(1)
        self.default_hidden_nonlinearity = tf.nn.tanh
        self.default_recurrent_nonlinearity = tf.nn.sigmoid
        self.default_output_nonlinearity = None
        self.time_step = 1

        self.policy1 = CategoricalLSTMPolicy(
            env_spec=env.spec,
            hidden_dim=4,
            hidden_nonlinearity=self.default_hidden_nonlinearity,
            hidden_w_init=self.default_initializer,
            recurrent_nonlinearity=self.default_recurrent_nonlinearity,
            recurrent_w_x_init=self.default_initializer,
            recurrent_w_h_init=self.default_initializer,
            output_nonlinearity=self.default_output_nonlinearity,
            output_w_init=self.default_initializer,
            state_include_action=True,
            name='P1')
        self.policy2 = CategoricalLSTMPolicy(
            env_spec=env.spec,
            hidden_dim=4,
            hidden_nonlinearity=self.default_hidden_nonlinearity,
            hidden_w_init=self.default_initializer,
            recurrent_nonlinearity=self.default_recurrent_nonlinearity,
            recurrent_w_x_init=self.default_initializer,
            recurrent_w_h_init=self.default_initializer,
            output_nonlinearity=self.default_output_nonlinearity,
            output_w_init=tf.constant_initializer(2),
            state_include_action=True,
            name='P2')

        self.sess.run(tf.global_variables_initializer())

        self.policy3 = CategoricalLSTMPolicyWithModel(
            env_spec=env.spec,
            hidden_dim=4,
            hidden_nonlinearity=self.default_hidden_nonlinearity,
            hidden_w_init=self.default_initializer,
            recurrent_nonlinearity=self.default_recurrent_nonlinearity,
            recurrent_w_init=self.default_initializer,
            output_nonlinearity=self.default_output_nonlinearity,
            output_w_init=self.default_initializer,
            state_include_action=True,
            name='P3')
        self.policy4 = CategoricalLSTMPolicyWithModel(
            env_spec=env.spec,
            hidden_dim=4,
            hidden_nonlinearity=self.default_hidden_nonlinearity,
            hidden_w_init=self.default_initializer,
            recurrent_nonlinearity=self.default_recurrent_nonlinearity,
            recurrent_w_init=self.default_initializer,
            output_nonlinearity=self.default_output_nonlinearity,
            output_w_init=tf.constant_initializer(2),
            state_include_action=True,
            name='P4')

        self.policy1.reset()
        self.policy2.reset()
        self.policy3.reset()
        self.policy4.reset()
        self.obs = [env.reset()]
        self.obs = np.concatenate([self.obs for _ in range(self.time_step)],
                                  axis=0)

        self.obs_ph = tf.placeholder(
            tf.float32, shape=(None, None, env.observation_space.flat_dim))
        self.action_ph = tf.placeholder(
            tf.float32, shape=(None, None, env.action_space.flat_dim))

        self.dist1_sym = self.policy1.dist_info_sym(
            obs_var=self.obs_ph,
            state_info_vars={'prev_action': np.zeros((2, self.time_step, 1))},
            name='p1_sym')
        self.dist2_sym = self.policy2.dist_info_sym(
            obs_var=self.obs_ph,
            state_info_vars={'prev_action': np.zeros((2, self.time_step, 1))},
            name='p2_sym')
        self.dist3_sym = self.policy3.dist_info_sym(
            obs_var=self.obs_ph,
            state_info_vars={'prev_action': np.zeros((2, self.time_step, 1))},
            name='p3_sym')
        self.dist4_sym = self.policy4.dist_info_sym(
            obs_var=self.obs_ph,
            state_info_vars={'prev_action': np.zeros((2, self.time_step, 1))},
            name='p4_sym')

    def test_dist_info_sym_output(self):
        # batch size = 2
        dist1 = self.sess.run(
            self.dist1_sym, feed_dict={self.obs_ph: [self.obs, self.obs]})
        dist2 = self.sess.run(
            self.dist2_sym, feed_dict={self.obs_ph: [self.obs, self.obs]})
        dist3 = self.sess.run(
            self.dist3_sym, feed_dict={self.obs_ph: [self.obs, self.obs]})
        dist4 = self.sess.run(
            self.dist4_sym, feed_dict={self.obs_ph: [self.obs, self.obs]})

        assert np.array_equal(dist1['prob'], dist3['prob'])
        assert np.array_equal(dist2['prob'], dist4['prob'])

    @mock.patch('numpy.random.rand')
    def test_get_action(self, mock_rand):
        mock_rand.return_value = 0

        action1, agent_info1 = self.policy1.get_action(self.obs)
        action2, agent_info2 = self.policy2.get_action(self.obs)
        action3, agent_info3 = self.policy3.get_action(self.obs)
        action4, agent_info4 = self.policy4.get_action(self.obs)

        assert action1 == action3
        assert action2 == action4
        assert np.array_equal(agent_info1['prob'], agent_info3['prob'])
        assert np.array_equal(agent_info2['prob'], agent_info4['prob'])

        actions1, agent_infos1 = self.policy1.get_actions([self.obs])
        actions2, agent_infos2 = self.policy2.get_actions([self.obs])
        actions3, agent_infos3 = self.policy3.get_actions([self.obs])
        actions4, agent_infos4 = self.policy4.get_actions([self.obs])

        assert np.array_equal(actions1, actions3)
        assert np.array_equal(actions2, actions4)
        assert np.array_equal(agent_infos1['prob'], agent_infos3['prob'])
        assert np.array_equal(agent_infos2['prob'], agent_infos4['prob'])

    def test_kl_sym(self):
        kl_diff_sym1 = self.policy1.distribution.kl_sym(
            self.dist1_sym, self.dist2_sym)
        objective1 = tf.reduce_mean(kl_diff_sym1)

        kl_func = tensor_utils.compile_function([self.obs_ph], objective1)
        kl1 = kl_func([self.obs, self.obs])

        kl_diff_sym2 = self.policy3.distribution.kl_sym(
            self.dist3_sym, self.dist4_sym)
        objective2 = tf.reduce_mean(kl_diff_sym2)

        kl_func = tensor_utils.compile_function([self.obs_ph], objective2)
        kl2 = kl_func([self.obs, self.obs])

        assert np.array_equal(kl1, kl2)

    def test_log_likehihood_sym(self):
        log_prob_sym1 = self.policy1.distribution.log_likelihood_sym(
            self.action_ph, self.dist1_sym)
        log_prob_func = tensor_utils.compile_function(
            [self.obs_ph, self.action_ph], log_prob_sym1)
        log_prob1 = log_prob_func([self.obs, self.obs],
                                  np.ones((2, self.time_step, 1)))

        log_prob_sym2 = self.policy3.distribution.log_likelihood_sym(
            self.action_ph, self.dist3_sym)
        log_prob_func2 = tensor_utils.compile_function(
            [self.obs_ph, self.action_ph], log_prob_sym2)
        log_prob2 = log_prob_func2([self.obs, self.obs],
                                   np.ones((2, self.time_step, 1)))
        assert np.array_equal(log_prob1, log_prob2)

        log_prob_sym1 = self.policy2.distribution.log_likelihood_sym(
            self.action_ph, self.dist2_sym)
        log_prob_func = tensor_utils.compile_function(
            [self.obs_ph, self.action_ph], log_prob_sym1)
        log_prob1 = log_prob_func([self.obs, self.obs],
                                  np.ones((2, self.time_step, 1)))

        log_prob_sym2 = self.policy4.distribution.log_likelihood_sym(
            self.action_ph, self.dist4_sym)
        log_prob_func2 = tensor_utils.compile_function(
            [self.obs_ph, self.action_ph], log_prob_sym2)
        log_prob2 = log_prob_func2([self.obs, self.obs],
                                   np.ones((2, self.time_step, 1)))
        assert np.array_equal(log_prob1, log_prob2)

    def test_policy_entropy_sym(self):
        entropy_sym1 = self.policy1.distribution.entropy_sym(
            self.dist1_sym, name='entropy_sym1')
        entropy_func = tensor_utils.compile_function([self.obs_ph],
                                                     entropy_sym1)
        entropy1 = entropy_func([self.obs, self.obs])

        entropy_sym2 = self.policy3.distribution.entropy_sym(
            self.dist3_sym, name='entropy_sym1')
        entropy_func = tensor_utils.compile_function([self.obs_ph],
                                                     entropy_sym2)
        entropy2 = entropy_func([self.obs, self.obs])
        assert np.array_equal(entropy1, entropy2)

    def test_likelihood_ratio_sym(self):
        likelihood_ratio_sym1 = self.policy1.distribution.likelihood_ratio_sym(
            self.action_ph,
            self.dist1_sym,
            self.dist2_sym,
            name='li_ratio_sym1')
        likelihood_ratio_func = tensor_utils.compile_function(
            [self.action_ph, self.obs_ph], likelihood_ratio_sym1)
        likelihood_ratio1 = likelihood_ratio_func(
            np.ones((2, 1, 1)), [self.obs, self.obs])

        likelihood_ratio_sym2 = self.policy3.distribution.likelihood_ratio_sym(
            self.action_ph,
            self.dist3_sym,
            self.dist4_sym,
            name='li_ratio_sym2')
        likelihood_ratio_func = tensor_utils.compile_function(
            [self.action_ph, self.obs_ph], likelihood_ratio_sym2)
        likelihood_ratio2 = likelihood_ratio_func(
            np.ones((2, 1, 1)), [self.obs, self.obs])

        assert np.array_equal(likelihood_ratio1, likelihood_ratio2)
