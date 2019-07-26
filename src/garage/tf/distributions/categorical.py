import numpy as np
import tensorflow as tf

from garage.tf.distributions.base import Distribution
from garage.tf.misc.tensor_utils import compile_function

TINY = 1e-8


def from_onehot(x_var):
    ret = np.zeros((len(x_var), ), 'int32')
    nonzero_n, nonzero_a = np.nonzero(x_var)
    ret[nonzero_n] = nonzero_a
    return ret


class Categorical(Distribution):
    def __init__(self, dim, name=None):
        with tf.compat.v1.variable_scope(name, 'Categorical'):
            self._dim = dim
            self._name = name
            weights_var = tf.compat.v1.placeholder(
                dtype=tf.float32, shape=(None, dim), name='weights')
            self._f_sample = compile_function(
                inputs=[weights_var],
                outputs=tf.random.categorical(
                    tf.math.log(weights_var + 1e-8), num_samples=1)[:, 0],
            )

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
            ndims = old_prob_var.get_shape().ndims
            # Assume layout is N * A
            return tf.reduce_sum(
                old_prob_var * (tf.math.log(old_prob_var + TINY) -
                                tf.math.log(new_prob_var + TINY)),
                axis=ndims - 1)

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two categorical distributions
        """
        old_prob = old_dist_info['prob']
        new_prob = new_dist_info['prob']
        return np.sum(
            old_prob * (np.log(old_prob + TINY) - np.log(new_prob + TINY)),
            axis=-1)

    def likelihood_ratio_sym(self,
                             x_var,
                             old_dist_info_vars,
                             new_dist_info_vars,
                             name=None):
        with tf.name_scope(name, 'likelihood_ratio_sym',
                           [x_var, old_dist_info_vars, new_dist_info_vars]):
            old_prob_var = old_dist_info_vars['prob']
            new_prob_var = new_dist_info_vars['prob']
            ndims = old_prob_var.get_shape().ndims
            x_var = tf.cast(x_var, tf.float32)
            # Assume layout is N * A
            return (tf.reduce_sum(new_prob_var * x_var, ndims - 1) + TINY) / \
                   (tf.reduce_sum(old_prob_var * x_var, ndims - 1) + TINY)

    def entropy_sym(self, dist_info_vars, name=None):
        with tf.name_scope(name, 'entropy_sym', [dist_info_vars]):
            probs = dist_info_vars['prob']
            return -tf.reduce_sum(probs * tf.math.log(probs + TINY), axis=1)

    def cross_entropy_sym(self,
                          old_dist_info_vars,
                          new_dist_info_vars,
                          name=None):
        with tf.name_scope(name, 'cross_entropy_sym',
                           [old_dist_info_vars, new_dist_info_vars]):
            old_prob_var = old_dist_info_vars['prob']
            new_prob_var = new_dist_info_vars['prob']
            ndims = old_prob_var.get_shape().ndims
            # Assume layout is N * A
            return tf.reduce_sum(
                old_prob_var * (-tf.math.log(new_prob_var + TINY)),
                axis=ndims - 1)

    def entropy(self, info):
        probs = info['prob']
        return -np.sum(probs * np.log(probs + TINY), axis=-1)

    def log_likelihood_sym(self, x_var, dist_info_vars, name=None):
        with tf.name_scope(name, 'log_likelihood_sym',
                           [x_var, dist_info_vars]):
            probs = dist_info_vars['prob']
            ndims = probs.get_shape().ndims
            return tf.math.log(
                tf.reduce_sum(probs * tf.cast(x_var, tf.float32), ndims - 1) +
                TINY)

    def log_likelihood(self, xs, dist_info):
        probs = dist_info['prob']
        # Assume layout is N * A
        return np.log(np.sum(probs * xs, axis=-1) + TINY)

    @property
    def dist_info_specs(self):
        return [('prob', (self.dim, ))]

    def sample(self, dist_info):
        return self._f_sample(dist_info['prob'])

    def sample_sym(self, dist_info, name=None):
        with tf.name_scope(name, 'sample_sym', [dist_info]):
            probs = dist_info['prob']
            samples = tf.multinomial(
                tf.math.log(probs + 1e-8), num_samples=1)[:, 0]

            return tf.nn.embedding_lookup(
                np.eye(self.dim, dtype=np.float32), samples)
