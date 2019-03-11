"""
Unit test for GaussianMLPPolicy.

This test consists of four different GaussianMLPPolicy: P1, P2, P3
and P4. All four policies are implemented with GaussianMLPModel.
P1 and P2 are from GaussianMLPPolicyWithModel, which uses
garage.tf.distributions while P3 and P4 from GaussianMLPPolicyWithModel2,
which uses tfp.distributions.

It does not aim to show how GaussianMLPModel will be used in GaussianMLPPolicy,
which is self-explanatory in the implementation of GaussianMLPPolicyWithModel.
It aims to illustrate the differences in operations between
GaussianMLPPolicyWithModel and GaussianMLPPolicyWithModel2.

It covers dist_info_sym, kl_sym, log_likelihood_sym, entropy_sym and
likelihood_ratio_sym. Also, it shows how a GaussianMLPPolicy should be pickled.
"""
import pickle

import numpy as np
import tensorflow as tf

from garage.tf.envs import TfEnv
from garage.tf.misc import tensor_utils
from garage.tf.policies import GaussianMLPPolicyWithModel
from garage.tf.policies import GaussianMLPPolicyWithModel2
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestGaussianMLPPolicyWithModel(TfGraphTestCase):
    def setUp(self):
        super().setUp()
        self.box_env = TfEnv(DummyBoxEnv())
        self.policy1 = GaussianMLPPolicyWithModel(
            env_spec=self.box_env, init_std=1.1, name='P1')
        self.policy2 = GaussianMLPPolicyWithModel(
            env_spec=self.box_env, init_std=1.2, name='P2')
        self.policy3 = GaussianMLPPolicyWithModel2(
            env_spec=self.box_env, init_std=1.1, name='P3')
        self.policy4 = GaussianMLPPolicyWithModel2(
            env_spec=self.box_env, init_std=1.2, name='P4')

        self.obs = [self.box_env.reset()]
        self.obs_ph = tf.placeholder(tf.float32, shape=(None, 1))

        # zero-mean data for testing
        self.zeros_obs = [np.zeros(1)]

        # API that remains unchanged
        # dist1_sym and dist2_sym are still dict(mean=mean, log_std=log_std)
        self.dist1_sym = self.policy1.dist_info_sym(self.obs_ph, name='p1_sym')
        self.dist2_sym = self.policy2.dist_info_sym(self.obs_ph, name='p2_sym')

        # this will be tf.distributions
        self.policy3.dist_info_sym(self.obs_ph, name='p3_sym')
        self.policy4.dist_info_sym(self.obs_ph, name='p4_sym')

        assert self.policy1.vectorized == self.policy2.vectorized
        assert self.policy3.vectorized == self.policy4.vectorized

    def test_get_action(self):
        action1, _ = self.policy1.get_action(self.obs)
        action2, _ = self.policy2.get_action(self.obs)
        action3 = self.policy3.get_action(self.obs)
        action4 = self.policy4.get_action(self.obs)

        assert self.box_env.action_space.contains(np.array(action1[0]))
        assert self.box_env.action_space.contains(np.array(action2[0]))
        assert self.box_env.action_space.contains(np.array(action3[0]))
        assert self.box_env.action_space.contains(np.array(action4[0]))

        actions1, _ = self.policy1.get_actions(self.obs)
        actions2, _ = self.policy2.get_actions(self.obs)
        actions3 = self.policy3.get_actions(self.obs)
        actions4 = self.policy4.get_actions(self.obs)

        assert self.box_env.action_space.contains(np.array(actions1[0]))
        assert self.box_env.action_space.contains(np.array(actions2[0]))
        assert self.box_env.action_space.contains(np.array(actions3[0]))
        assert self.box_env.action_space.contains(np.array(actions4[0]))

    def test_kl_sym(self):
        # kl_sym
        # * Existing way *
        kl_diff_sym1 = self.policy1.distribution.kl_sym(
            self.dist1_sym, self.dist2_sym)
        objective1 = tf.reduce_mean(kl_diff_sym1)

        kl_func = tensor_utils.compile_function([self.obs_ph], objective1)
        kl1 = kl_func(self.zeros_obs, self.zeros_obs)

        # * New way *
        # equvaient to distribution.kl_sym()
        kl_diff_sym2 = self.policy3.model.networks[
            'default'].dist.kl_divergence(
                self.policy4.model.networks['default'].dist)
        objective2 = tf.reduce_mean(kl_diff_sym2)

        kl2 = self.sess.run(
            objective2,
            feed_dict={
                self.policy3.model.networks['default'].input: self.zeros_obs,
                self.policy4.model.networks['default'].input: self.zeros_obs
            })

        self.assertAlmostEqual(kl1, kl2)

    def test_log_likehihood_sym(self):
        # log_likelihood_sym
        # * Existing way *
        log_prob_sym1 = self.policy1.distribution.log_likelihood_sym(
            self.obs_ph, self.dist1_sym)
        log_prob_func = tensor_utils.compile_function([self.obs_ph],
                                                      log_prob_sym1)
        log_prob1 = log_prob_func(self.zeros_obs)

        # * New way *
        # equvaient to distribution.log_likelihood_sym(X, dist)
        log_prob_sym2 = self.policy3.model.networks['default'].dist.log_prob(
            self.policy3.model.networks['default'].input)
        log_prob2 = self.sess.run(  # result
            log_prob_sym2,
            feed_dict={
                self.policy3.model.networks['default'].input: self.zeros_obs
            })

        assert log_prob1 == log_prob2

    def test_policy_entropy_sym(self):
        # entropy_sym
        # * Existing way *
        entropy_sym1 = self.policy1.distribution.entropy_sym(
            self.dist1_sym, name='entropy_sym1')
        entropy_func = tensor_utils.compile_function([self.obs_ph],
                                                     entropy_sym1)
        entropy1 = entropy_func(self.zeros_obs)  # result

        # * New way *
        # equvaient to garage.tf.distribution.entropy_sym(dist)
        entropy_sym2 = self.policy3.model.networks['default'].dist.entropy()
        entropy2 = self.sess.run(  # result
            entropy_sym2,
            feed_dict={
                self.policy3.model.networks['default'].input: self.zeros_obs
            })

        assert entropy1 == entropy2

    def test_likehihood_sym(self):
        # likelihood_ratio_sym
        # * Existing way *
        likelihood_ratio_sym1 = self.policy1.distribution.likelihood_ratio_sym(
            self.obs_ph, self.dist1_sym, self.dist2_sym, name='li_ratio_sym1')
        likelihood_ratio_func = tensor_utils.compile_function(
            [self.obs_ph], likelihood_ratio_sym1)
        likelihood_ratio1 = likelihood_ratio_func(self.zeros_obs)

        # * New way *
        with tf.name_scope('li_ratio_sym2'):
            likelihood_ratio_sym2 = self.policy4.likelihood_ratio_sym(
                self.obs_ph, self.policy3.model.networks['default'].dist)
            likelihood_ratio2 = self.sess.run(  # result
                likelihood_ratio_sym2,
                feed_dict={
                    self.policy3.model.networks['default'].input:
                        self.zeros_obs,
                    self.policy4.model.networks['default'].input:
                        self.zeros_obs,
                    self.obs_ph: self.zeros_obs
                })

        assert likelihood_ratio1 == likelihood_ratio2

        # input with wrong shape will raise error
        obs_ph2 = tf.placeholder(tf.float32, shape=(None, 10))
        with self.assertRaises(AssertionError):
            self.policy4.likelihood_ratio_sym(
                obs_ph2, self.policy3.model.networks['default'].dist)

    def test_is_pickleable(self):
        with tf.Session(graph=tf.Graph()) as sess:
            policy = GaussianMLPPolicyWithModel(env_spec=self.box_env)
            # model is built in GaussianMLPPolicyWithModel.__init__
            outputs = sess.run(
                policy.model.networks['default'].sample,
                feed_dict={policy.model.networks['default'].input: self.obs})
            p = pickle.dumps(policy)

        with tf.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            outputs2 = sess.run(
                policy_pickled.model.networks['default'].sample,
                feed_dict={
                    policy_pickled.model.networks['default'].input: self.obs
                })

        assert np.array_equal(outputs, outputs2)

    def test_is_pickleable2(self):
        with tf.Session(graph=tf.Graph()) as sess:
            policy = GaussianMLPPolicyWithModel2(env_spec=self.box_env)
            # model is built in GaussianMLPPolicyWithModel2.__init__
            outputs = sess.run(
                policy.model.networks['default'].sample,
                feed_dict={policy.model.networks['default'].input: self.obs})
            p = pickle.dumps(policy)

        with tf.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            # After pickle, we need to build the model
            # e.g. by policy.dist_info_sym
            input_ph = self.box_env.observation_space.new_tensor_variable(
                extra_dims=1, name='input_ph')
            policy_pickled.dist_info_sym(input_ph)

            outputs2 = sess.run(
                policy_pickled.model.networks['default'].sample,
                feed_dict={
                    policy_pickled.model.networks['default'].input: self.obs
                })

        assert np.array_equal(outputs, outputs2)
