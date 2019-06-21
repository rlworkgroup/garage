"""
Unit test for GaussianMLPPolicyWithModel.

This test consists of four different GaussianMLPPolicy: P1, P2, P3
and P4. P1 and P2 are from GaussianMLPPolicy, which does not use
garage.tf.models.GaussianMLPModel while P3 and P4 do use.

This test ensures the outputs from all the policies are the same,
for the transition from using GaussianMLPPolicy to
GaussianMLPPolicyWithModel.

It covers get_action, get_actions, dist_info_sym, kl_sym,
log_likelihood_sym, entropy_sym and likelihood_ratio_sym.
"""
from unittest import mock

import numpy as np
import pytest
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.misc import tensor_utils
from garage.tf.policies import GaussianMLPPolicy
from garage.tf.policies import GaussianMLPPolicyWithModel
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestGaussianMLPPolicyWithModelTransit(TfGraphTestCase):
    def setup_method(self):
        with mock.patch('tensorflow.random.normal') as mock_rand:
            mock_rand.return_value = 0.5
            super().setup_method()
            self.box_env = TfEnv(DummyBoxEnv())
            self.policy1 = GaussianMLPPolicy(
                env_spec=self.box_env, init_std=1.0, name='P1')
            self.policy2 = GaussianMLPPolicy(
                env_spec=self.box_env, init_std=1.2, name='P2')
            self.policy3 = GaussianMLPPolicyWithModel(
                env_spec=self.box_env, init_std=1.0, name='P3')
            self.policy4 = GaussianMLPPolicyWithModel(
                env_spec=self.box_env, init_std=1.2, name='P4')

            self.sess.run(tf.global_variables_initializer())

            for a, b in zip(self.policy3.get_params(),
                            self.policy1.get_params()):
                self.sess.run(tf.assign(b, a))
            for a, b in zip(self.policy4.get_params(),
                            self.policy2.get_params()):
                self.sess.run(tf.assign(b, a))

            self.obs = [self.box_env.reset()]
            self.obs_ph = tf.placeholder(
                tf.float32,
                shape=(None, self.box_env.observation_space.flat_dim))
            self.action_ph = tf.placeholder(
                tf.float32, shape=(None, self.box_env.action_space.flat_dim))

            self.dist1_sym = self.policy1.dist_info_sym(
                self.obs_ph, name='p1_sym')
            self.dist2_sym = self.policy2.dist_info_sym(
                self.obs_ph, name='p2_sym')
            self.dist3_sym = self.policy3.dist_info_sym(
                self.obs_ph, name='p3_sym')
            self.dist4_sym = self.policy4.dist_info_sym(
                self.obs_ph, name='p4_sym')

            assert self.policy1.vectorized == self.policy2.vectorized
            assert self.policy3.vectorized == self.policy4.vectorized

    def test_dist_info_sym_output(self):
        dist1 = self.sess.run(
            self.dist1_sym, feed_dict={self.obs_ph: self.obs})
        dist2 = self.sess.run(
            self.dist2_sym, feed_dict={self.obs_ph: self.obs})
        dist3 = self.sess.run(
            self.dist3_sym, feed_dict={self.obs_ph: self.obs})
        dist4 = self.sess.run(
            self.dist4_sym, feed_dict={self.obs_ph: self.obs})

        assert np.array_equal(dist1['mean'], dist3['mean'])
        assert np.array_equal(dist1['log_std'], dist3['log_std'])
        assert np.array_equal(dist2['mean'], dist4['mean'])
        assert np.array_equal(dist2['log_std'], dist4['log_std'])

    @mock.patch('numpy.random.normal')
    def test_get_action(self, mock_rand):
        mock_rand.return_value = 0.5
        action1, _ = self.policy1.get_action(self.obs)
        action2, _ = self.policy2.get_action(self.obs)
        action3, _ = self.policy3.get_action(self.obs)
        action4, _ = self.policy4.get_action(self.obs)

        assert np.array_equal(action1, action3)
        assert np.array_equal(action2, action4)

        actions1, dist_info1 = self.policy1.get_actions([self.obs])
        actions2, dist_info2 = self.policy2.get_actions([self.obs])
        actions3, dist_info3 = self.policy3.get_actions([self.obs])
        actions4, dist_info4 = self.policy4.get_actions([self.obs])

        assert np.array_equal(actions1, actions3)
        assert np.array_equal(actions2, actions4)

        assert np.array_equal(dist_info1['mean'], dist_info3['mean'])
        assert np.array_equal(dist_info1['log_std'], dist_info3['log_std'])
        assert np.array_equal(dist_info2['mean'], dist_info4['mean'])
        assert np.array_equal(dist_info2['log_std'], dist_info4['log_std'])

    def test_kl_sym(self):
        kl_diff_sym1 = self.policy1.distribution.kl_sym(
            self.dist1_sym, self.dist2_sym)
        objective1 = tf.reduce_mean(kl_diff_sym1)

        kl_func = tensor_utils.compile_function([self.obs_ph], objective1)
        kl1 = kl_func(self.obs, self.obs)

        kl_diff_sym2 = self.policy3.distribution.kl_sym(
            self.dist3_sym, self.dist4_sym)
        objective2 = tf.reduce_mean(kl_diff_sym2)

        kl_func = tensor_utils.compile_function([self.obs_ph], objective2)
        kl2 = kl_func(self.obs, self.obs)

        assert np.array_equal(kl1, kl2)
        assert kl1 == pytest.approx(kl2)

    def test_log_likehihood_sym(self):
        log_prob_sym1 = self.policy1.distribution.log_likelihood_sym(
            self.action_ph, self.dist1_sym)
        log_prob_func = tensor_utils.compile_function(
            [self.obs_ph, self.action_ph], log_prob_sym1)
        log_prob1 = log_prob_func(self.obs, [[1, 1]])

        log_prob_sym2 = self.policy3.model.networks[
            'default'].dist.log_likelihood_sym(self.action_ph, self.dist3_sym)
        log_prob_func2 = tensor_utils.compile_function(
            [self.obs_ph, self.action_ph], log_prob_sym2)
        log_prob2 = log_prob_func2(self.obs, [[1, 1]])
        assert log_prob1 == log_prob2

        log_prob_sym1 = self.policy2.distribution.log_likelihood_sym(
            self.action_ph, self.dist2_sym)
        log_prob_func = tensor_utils.compile_function(
            [self.obs_ph, self.action_ph], log_prob_sym1)
        log_prob1 = log_prob_func(self.obs, [[1, 1]])

        log_prob_sym2 = self.policy4.model.networks[
            'default'].dist.log_likelihood_sym(self.action_ph, self.dist4_sym)
        log_prob_func2 = tensor_utils.compile_function(
            [self.obs_ph, self.action_ph], log_prob_sym2)
        log_prob2 = log_prob_func2(self.obs, [[1, 1]])
        assert log_prob1 == log_prob2

    def test_policy_entropy_sym(self):
        entropy_sym1 = self.policy1.distribution.entropy_sym(
            self.dist1_sym, name='entropy_sym1')
        entropy_func = tensor_utils.compile_function([self.obs_ph],
                                                     entropy_sym1)
        entropy1 = entropy_func(self.obs)

        entropy_sym2 = self.policy3.distribution.entropy_sym(
            self.dist3_sym, name='entropy_sym1')
        entropy_func = tensor_utils.compile_function([self.obs_ph],
                                                     entropy_sym2)
        entropy2 = entropy_func(self.obs)
        assert entropy1 == entropy2

    def test_likelihood_ratio_sym(self):
        likelihood_ratio_sym1 = self.policy1.distribution.likelihood_ratio_sym(
            self.action_ph,
            self.dist1_sym,
            self.dist2_sym,
            name='li_ratio_sym1')
        likelihood_ratio_func = tensor_utils.compile_function(
            [self.action_ph, self.obs_ph], likelihood_ratio_sym1)
        likelihood_ratio1 = likelihood_ratio_func([[1, 1]], self.obs)

        likelihood_ratio_sym2 = self.policy3.distribution.likelihood_ratio_sym(
            self.action_ph,
            self.dist3_sym,
            self.dist4_sym,
            name='li_ratio_sym2')
        likelihood_ratio_func = tensor_utils.compile_function(
            [self.action_ph, self.obs_ph], likelihood_ratio_sym2)
        likelihood_ratio2 = likelihood_ratio_func([[1, 1]], self.obs)

        assert likelihood_ratio1 == likelihood_ratio2
