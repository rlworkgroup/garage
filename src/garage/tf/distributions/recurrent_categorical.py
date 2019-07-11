import numpy as np
import tensorflow as tf

from garage.tf.distributions.base import Distribution
from garage.tf.distributions.categorical import Categorical

TINY = 1e-8


class RecurrentCategorical(Distribution):
    def __init__(self, dim, name='RecurrentCategorical'):
        self._cat = Categorical(dim, name)
        self._dim = dim
        self._name = name

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars, name=None):
        """
        Compute the symbolic KL divergence of two categorical distributions
        """
        with tf.name_scope(name, 'kl_sym',
                           [old_dist_info_vars, new_dist_info_vars]):
            old_prob_var = old_dist_info_vars['prob']
            new_prob_var = new_dist_info_vars['prob']
            # Assume layout is N * T * A
            return tf.reduce_sum(
                old_prob_var * (tf.math.log(old_prob_var + TINY) -
                                tf.math.log(new_prob_var + TINY)),
                axis=2)

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two categorical distributions
        """
        old_prob = old_dist_info['prob']
        new_prob = new_dist_info['prob']
        return np.sum(
            old_prob * (np.log(old_prob + TINY) - np.log(new_prob + TINY)),
            axis=2)

    def likelihood_ratio_sym(self,
                             x_var,
                             old_dist_info_vars,
                             new_dist_info_vars,
                             name=None):
        with tf.name_scope(name, 'likelihood_ratio_sym',
                           [x_var, old_dist_info_vars, new_dist_info_vars]):
            old_prob_var = old_dist_info_vars['prob']
            new_prob_var = new_dist_info_vars['prob']
            # Assume layout is N * T * A
            a_dim = tf.shape(x_var)[2]
            flat_ratios = self._cat.likelihood_ratio_sym(
                tf.reshape(x_var, tf.stack([-1, a_dim])),
                dict(prob=tf.reshape(old_prob_var, tf.stack([-1, a_dim]))),
                dict(prob=tf.reshape(new_prob_var, tf.stack([-1, a_dim]))))
            return tf.reshape(flat_ratios, tf.shape(old_prob_var)[:2])

    def entropy(self, dist_info):
        probs = dist_info['prob']
        return -np.sum(probs * np.log(probs + TINY), axis=2)

    def entropy_sym(self, dist_info_vars, name=None):
        with tf.name_scope(name, 'entropy_sym', [dist_info_vars]):
            probs = dist_info_vars['prob']
            return -tf.reduce_sum(probs * tf.math.log(probs + TINY), 2)

    def log_likelihood_sym(self, xs, dist_info_vars, name=None):
        with tf.name_scope(name, 'log_likelihood_sym', [xs, dist_info_vars]):
            probs = dist_info_vars['prob']
            # Assume layout is N * T * A
            a_dim = tf.shape(probs)[2]
            flat_logli = self._cat.log_likelihood_sym(
                tf.reshape(xs, tf.stack([-1, a_dim])),
                dict(prob=tf.reshape(probs, tf.stack((-1, a_dim)))))
            return tf.reshape(flat_logli, tf.shape(probs)[:2])

    def log_likelihood(self, xs, dist_info):
        probs = dist_info['prob']
        # Assume layout is N * T * A
        a_dim = tf.shape(probs)[2]
        flat_logli = self._cat.log_likelihood_sym(
            xs.reshape((-1, a_dim)), dict(prob=probs.reshape((-1, a_dim))))
        return flat_logli.reshape(probs.shape[:2])

    @property
    def dist_info_specs(self):
        return [('prob', (self.dim, ))]
