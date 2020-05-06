"""Diagonal Gaussian Distribution."""
import numpy as np
import tensorflow as tf

from garage.tf.distributions.distribution import Distribution


class DiagonalGaussian(Distribution):
    """Diagonal Gaussian Distribution.

    Args:
        dim (int): Dimension of the distribution.
        name (str): Name (scope) of the distribution.

    """

    def __init__(self, dim, name='DiagonalGaussian'):
        self._dim = dim
        self._name = name

    @property
    def dim(self):
        """int: Dimension of the distribution."""
        return self._dim

    def kl(self, old_dist_info, new_dist_info):
        """KL Divergence between the old and the new distribution.

        Args:
            old_dist_info (dict): Parameters of the old distribution.
            new_dist_info (dict): Parameters of the new distribution.

        Returns:
            float: KL Divergence between two distributions.

        """
        old_means = old_dist_info['mean']
        old_log_stds = old_dist_info['log_std']
        new_means = new_dist_info['mean']
        new_log_stds = new_dist_info['log_std']
        # Compute the KL divergence of two multivariate Gaussian distribution
        # with diagonal covariance matrices
        old_std = np.exp(old_log_stds)
        new_std = np.exp(new_log_stds)
        # means: (N*A)
        # std: (N*A)
        # formula:
        # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
        # ln(\sigma_2/\sigma_1)
        numerator = np.square(old_means - new_means) + \
            np.square(old_std) - np.square(new_std)
        denominator = 2 * np.square(new_std) + 1e-8
        return np.sum(numerator / denominator + new_log_stds - old_log_stds,
                      axis=-1)

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars, name='kl_sym'):
        """Symbolic KL between the old and the new distribution.

        Args:
            old_dist_info_vars (tf.Tensor): Symbolic parameters of
                the old distribution.
            new_dist_info_vars (tf.Tensor): Symbolic parameters of
                the new distribution.
            name (str): TensorFlow scope name.

        Returns:
            tf.Tensor: Symbolic KL divergence between the two distributions.

        """
        with tf.name_scope(name):
            old_means = old_dist_info_vars['mean']
            old_log_stds = old_dist_info_vars['log_std']
            new_means = new_dist_info_vars['mean']
            new_log_stds = new_dist_info_vars['log_std']
            # Compute the KL divergence of two multivariate
            # Gaussian distribution with diagonal covariance matrices
            old_std = tf.exp(old_log_stds)
            new_std = tf.exp(new_log_stds)
            # means: (N*A)
            # std: (N*A)
            # formula:
            # { (\mu_1 - \mu_2)^2 + \sigma_1^2 - \sigma_2^2 } / (2\sigma_2^2) +
            # ln(\sigma_2/\sigma_1)
            numerator = tf.square(old_means - new_means) + \
                tf.square(old_std) - tf.square(new_std)
            denominator = 2 * tf.square(new_std) + 1e-8
            return tf.reduce_sum(numerator / denominator + new_log_stds -
                                 old_log_stds,
                                 axis=-1)

    def likelihood_ratio_sym(self,
                             x_var,
                             old_dist_info_vars,
                             new_dist_info_vars,
                             name='likelihood_ratio_sym'):
        """Symbolic likelihood ratio.

        Args:
            x_var (tf.Tensor): Input placeholder.
            old_dist_info_vars (dict): Old distribution tensors.
            new_dist_info_vars (dict): New distribution tensors.
            name (str): TensorFlow scope name.

        Returns:
            tf.Tensor: Symbolic likelihood ratio.

        """
        with tf.name_scope(name):
            logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
            logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
            return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self,
                           x_var,
                           dist_info_vars,
                           name='log_likelihood_sym'):
        """Symbolic log likelihood.

        Args:
            x_var (tf.Tensor): Input placeholder.
            dist_info_vars (dict): Parameters of a distribution.
            name (str): TensorFlow scope name.

        Returns:
            tf.Tensor: Symbolic log likelihood.

        """
        with tf.name_scope(name):
            means = dist_info_vars['mean']
            log_stds = dist_info_vars['log_std']
            zs = (x_var - means) / tf.exp(log_stds)
            return - tf.reduce_sum(log_stds, axis=-1) - \
                0.5 * tf.reduce_sum(tf.square(zs), axis=-1) - \
                0.5 * self.dim * np.log(2 * np.pi)

    def sample(self, dist_info):
        """Sample a value given a distribution.

        Args:
            dist_info (dict): Parameters of a distribution.

        Returns:
            np.ndarray: A sample from the distribution.

        """
        # pylint: disable=no-self-use
        means = dist_info['mean']
        log_stds = dist_info['log_std']
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    def sample_sym(self, dist_info_vars):
        """Sample a symbolic value given a distribution.

        Args:
            dist_info_vars (dict): Symbolic parameters of a distribution.

        Returns:
            tf.Tensor: A symbolic sample from the distribution.

        """
        # pylint: disable=no-self-use
        means = dist_info_vars['mean']
        log_stds = dist_info_vars['log_std']
        rnd = tf.random.normal(shape=tf.shape(means))
        return rnd * tf.math.exp(log_stds) + means

    def log_likelihood(self, xs, dist_info):
        """Log likelihood of a sample under a distribution.

        Args:
            xs (np.ndarray): Input value.
            dist_info (dict): Parameters of a distribution.

        Returns:
            float: Log likelihood of a sample under the distribution.

        """
        means = dist_info['mean']
        log_stds = dist_info['log_std']
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
            0.5 * np.sum(np.square(zs), axis=-1) - \
            0.5 * self.dim * np.log(2 * np.pi)

    def entropy(self, dist_info):
        """Entropy of a distribution.

        Args:
            dist_info (dict): Parameters of a distribution.

        Returns:
            float: Entropy of the distribution.

        """
        log_stds = dist_info['log_std']
        return np.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    def entropy_sym(self, dist_info_vars, name='entropy_sym'):
        """Symbolic entropy of a distribution.

        Args:
            dist_info_vars (dict): Symbolic parameters of a distribution.
            name (str): TensorFlow scope name.

        Returns:
            tf.Tensor: Symbolic entropy of the distribution.

        """
        with tf.name_scope(name):
            log_std_var = dist_info_vars['log_std']
            return tf.reduce_sum(log_std_var +
                                 np.log(np.sqrt(2 * np.pi * np.e)),
                                 axis=-1)

    @property
    def dist_info_specs(self):
        """list: Specification of the parameter of a distribution."""
        return [('mean', (self.dim, )), ('log_std', (self.dim, ))]
