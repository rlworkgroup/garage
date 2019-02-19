"""
Unit test for GaussianMLPPolicy.

This test consists of four different GaussianMLPPolicy: P1, P2, P3
and P4. All four policies are implemented with GaussianMLPModel.
P1 and P2 are from GaussianMLPPolicyWithModel, which uses
garage.tf.distributions while P3 and P4 from GaussianMLPPolicyWithModel2,
which uses tf.distributions.

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
import tensorflow_probability as tfp

from garage.tf.envs import TfEnv
from garage.tf.misc import tensor_utils
from garage.tf.policies import GaussianMLPPolicyWithModel
from garage.tf.policies import GaussianMLPPolicyWithModel2
from tests.fixtures import TfGraphTestCase
from tests.fixtures.envs.dummy import DummyBoxEnv


class TestGaussianMLPPolicyWithModel(TfGraphTestCase):
    def test_gaussian_mlp_policy_with_model(self):
        box_env = TfEnv(DummyBoxEnv())
        policy1 = GaussianMLPPolicyWithModel(
            env_spec=box_env, init_std=1.1, name='P1')
        policy2 = GaussianMLPPolicyWithModel(
            env_spec=box_env, init_std=1.2, name='P2')
        policy3 = GaussianMLPPolicyWithModel2(
            env_spec=box_env, init_std=1.1, name='P3')
        policy4 = GaussianMLPPolicyWithModel2(
            env_spec=box_env, init_std=1.2, name='P4')

        # self.sess.run(tf.global_variables_initializer())

        obs = [box_env.reset()]

        obs_ph = tf.placeholder(tf.float32, shape=(None, 1))

        ############################################################
        # dist_info_sym
        # API unchanged
        #
        # * Existing way *
        # dist1_sym and dist2_sym are still dict(mean=mean, log_std=log_std)
        dist1_sym = policy1.dist_info_sym(obs_ph, name='p1_sym')
        dist2_sym = policy2.dist_info_sym(obs_ph, name='p2_sym')

        # * New way *
        # this will be tf.distributions
        policy3.dist_info_sym(obs_ph, name='p3_sym')
        policy4.dist_info_sym(obs_ph, name='p4_sym')

        ############################################################
        # kl_sym
        # * Existing way *
        kl_diff_sym1 = policy1.distribution.kl_sym(dist1_sym, dist2_sym)
        objective1 = tf.reduce_mean(kl_diff_sym1)

        kl_func = tensor_utils.compile_function([obs_ph], objective1)
        kl1 = kl_func(obs, obs)

        # * New way *
        # equvaient to distribution.kl_sym()
        kl_diff_sym2 = tfp.distributions.kl_divergence(
            policy3.model.networks['default'].distribution,
            policy4.model.networks['default'].distribution)
        objective2 = tf.reduce_mean(kl_diff_sym2)

        kl2 = self.sess.run(
            objective2,
            feed_dict={
                policy3.model.networks['default'].input: obs,
                policy4.model.networks['default'].input: obs
            })

        self.assertAlmostEqual(kl1, kl2)

        ############################################################
        # log_likelihood_sym
        # * Existing way *
        log_prob_sym1 = policy1.distribution.log_likelihood_sym(
            obs_ph, dist1_sym)
        log_prob_func = tensor_utils.compile_function([obs_ph], log_prob_sym1)
        log_prob1 = log_prob_func(obs)

        # * New way *
        # equvaient to distribution.log_likelihood_sym(X, dist)
        log_prob_sym2 = policy3.model.networks[
            'default'].distribution.log_prob(
                policy3.model.networks['default'].input)
        log_prob2 = self.sess.run(  # result
            log_prob_sym2,
            feed_dict={policy3.model.networks['default'].input: obs})

        assert log_prob1 == log_prob2

        ############################################################
        # entropy_sym
        # * Existing way *
        entropy_sym1 = policy1.distribution.entropy_sym(
            dist1_sym, name='entropy_sym1')
        entropy_func = tensor_utils.compile_function([obs_ph], entropy_sym1)
        entropy1 = entropy_func(obs)  # result

        # * New way *
        # equvaient to garage.tf.distribution.entropy_sym(dist)
        entropy_sym2 = policy3.model.networks['default'].distribution.entropy()
        entropy2 = self.sess.run(  # result
            entropy_sym2,
            feed_dict={policy3.model.networks['default'].input: obs})

        assert entropy1 == entropy2

        ############################################################
        # likelihood_ratio_sym
        # * Existing way *
        likelihood_ratio_sym1 = policy1.distribution.likelihood_ratio_sym(
            obs_ph, dist1_sym, dist2_sym, name='li_ratio_sym1')
        likelihood_ratio_func = tensor_utils.compile_function(
            [obs_ph], likelihood_ratio_sym1)
        likelihood_ratio1 = likelihood_ratio_func(obs)

        # * New way *
        # tf.distributions seems doesn't have this available
        # maybe do it ourselves
        with tf.name_scope('li_ratio_sym2'):
            log_prob_diff = policy4.model.networks[
                'default'].distribution.log_prob(
                    obs_ph, name='log_prob_obs') - policy3.model.networks[
                        'default'].distribution.log_prob(obs_ph)

            likelihood_ratio_sym2 = tf.exp(log_prob_diff)
            likelihood_ratio2 = self.sess.run(  # result
                likelihood_ratio_sym2,
                feed_dict={
                    policy3.model.networks['default'].input: obs,
                    policy4.model.networks['default'].input: obs,
                    obs_ph: obs
                })

        assert likelihood_ratio1 == likelihood_ratio2

    def test_guassian_mlp_policy_pickle(self):
        box_env = TfEnv(DummyBoxEnv())
        data = np.ones((3, 1))
        with tf.Session(graph=tf.Graph()) as sess:
            policy = GaussianMLPPolicyWithModel2(
                env_spec=box_env, init_std=1.1)
            # model is built in GaussianMLPPolicyWithModel2.__init__
            outputs = sess.run(
                policy.model.networks['default'].sample,
                feed_dict={policy.model.networks['default'].input: data})
            p = pickle.dumps(policy)

        with tf.Session(graph=tf.Graph()) as sess:
            policy_pickled = pickle.loads(p)
            # After pickle, we need to build the model
            # e.g. by policy.dist_info_sym
            input_ph = box_env.observation_space.new_tensor_variable(
                extra_dims=1, name='input_ph')
            policy_pickled.dist_info_sym(
                input_ph, tfp.distributions.MultivariateNormalDiag)

            outputs2 = sess.run(
                policy_pickled.model.networks['default'].sample,
                feed_dict={
                    policy_pickled.model.networks['default'].input: data
                })

        assert np.array_equal(outputs, outputs2)
