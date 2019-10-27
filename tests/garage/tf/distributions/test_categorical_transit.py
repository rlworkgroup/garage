import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from garage.tf.distributions import Categorical
from tests.fixtures import TfGraphTestCase


class TestCategoricalTransit(TfGraphTestCase):

    def setup_method(self):
        super().setup_method()
        dim = 3
        self.garage_cat = Categorical(dim=dim)
        self.labels = np.array([1, 0, 0])
        self.prob1 = np.array([[0.5, 0.4, 0.1], [0.4, 0.3, 0.3],
                               [0.25, 0.74, 0.01]])
        self.prob2 = np.array([[0.2, 0.6, 0.2], [0.1, 0.7, 0.2],
                               [0.1, 0.44, 0.46]])
        self.old_prob_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                                     name='old_prob',
                                                     shape=(None, 3))
        self.prob_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                                 name='prob',
                                                 shape=(None, 3))
        self.tf_cat1 = tfp.distributions.OneHotCategorical(
            probs=self.old_prob_var)
        self.tf_cat2 = tfp.distributions.OneHotCategorical(probs=self.prob_var)
        self.labels_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                                   name='labels',
                                                   shape=(None, 3))

    def test_kl_sym(self):
        garage_kl_sym = self.garage_cat.kl_sym(dict(prob=self.old_prob_var),
                                               dict(prob=self.prob_var))
        kl_sym = self.sess.run(garage_kl_sym,
                               feed_dict={
                                   self.old_prob_var: self.prob1,
                                   self.prob_var: self.prob2
                               })

        kl_sym2 = self.sess.run(self.tf_cat1.kl_divergence(self.tf_cat2),
                                feed_dict={
                                    self.old_prob_var: self.prob1,
                                    self.prob_var: self.prob2
                                })
        assert np.allclose(kl_sym, kl_sym2)

    def test_log_prob_sym(self):
        log_prob = self.sess.run(self.garage_cat.log_likelihood_sym(
            self.labels_var, dict(prob=self.prob_var)),
                                 feed_dict={
                                     self.labels_var: [self.labels],
                                     self.prob_var: self.prob1
                                 })
        log_prob2 = self.sess.run(self.tf_cat1.log_prob(self.labels_var),
                                  feed_dict={
                                      self.labels_var: [self.labels],
                                      self.old_prob_var: self.prob1
                                  })
        assert np.allclose(log_prob, log_prob2)

    def test_entropy_sym(self):
        entropy = self.sess.run(self.garage_cat.entropy_sym(
            dict(prob=self.prob_var)),
                                feed_dict={self.prob_var: self.prob1})
        entropy2 = self.sess.run(self.tf_cat1.entropy(),
                                 feed_dict={self.old_prob_var: self.prob1})
        assert np.allclose(entropy, entropy2)

    def test_cross_entropy_sym(self):
        cross_entropy = self.sess.run(self.garage_cat.cross_entropy_sym(
            old_dist_info_vars=dict(prob=self.old_prob_var),
            new_dist_info_vars=dict(prob=self.prob_var)),
                                      feed_dict={
                                          self.old_prob_var: self.prob1,
                                          self.prob_var: self.prob2
                                      })
        cross_entropy2 = self.sess.run(self.tf_cat1.cross_entropy(
            self.tf_cat2),
                                       feed_dict={
                                           self.old_prob_var: self.prob1,
                                           self.prob_var: self.prob2
                                       })
        assert np.allclose(cross_entropy, cross_entropy2)

    def test_likelihood_ratio_sym(self):
        llr = self.sess.run(self.garage_cat.likelihood_ratio_sym(
            x_var=self.labels_var,
            old_dist_info_vars=dict(prob=self.old_prob_var),
            new_dist_info_vars=dict(prob=self.prob_var)),
                            feed_dict={
                                self.labels_var: [self.labels],
                                self.old_prob_var: self.prob1,
                                self.prob_var: self.prob2
                            })
        llr2 = self.sess.run(self.tf_cat2.prob(self.labels_var) /
                             self.tf_cat1.prob(self.labels_var),
                             feed_dict={
                                 self.labels_var: [self.labels],
                                 self.old_prob_var: self.prob1,
                                 self.prob_var: self.prob2
                             })
        assert np.allclose(llr, llr2)

    def test_sample_sym(self):
        sample = self.sess.run(self.garage_cat.sample_sym(
            dict(prob=self.prob_var)),
                               feed_dict={self.prob_var: self.prob1})
        sample2 = self.sess.run(self.tf_cat1.sample(),
                                feed_dict={self.old_prob_var: self.prob1})
        assert sample.shape == sample2.shape
