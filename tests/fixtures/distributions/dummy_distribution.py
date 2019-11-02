"""Dummy distribution for testing purpose."""
from garage.tf.distributions.base import Distribution


class DummyDistribution(Distribution):
    """Dummy distribution for testing purpose."""

    @property
    def dim(self):
        """Distribution dimension.

        Returns:
            int: Distribution dimenison.

        """
        return 1

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """Compute the symbolic KL divergence of two distributions.

        Args:
            old_dist_info_vars (dict): Old distribution tensors.
            new_dist_info_vars (dict): New distribution tensors.
        """

    def kl(self, old_dist_info, new_dist_info):
        """Compute the KL divergence of two distributions.

        Args:
            old_dist_info (dict): Old distribution parameters.
            new_dist_info (dict): New distribution parameters.
        """

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars,
                             new_dist_info_vars):
        """Likelihood ratio sym.

        Args:
            x_var (tf.Tensor): Input placeholder.
            old_dist_info_vars (dict): Old distribution tensors.
            new_dist_info_vars (dict): New distribution tensors.
        """

    def entropy(self, dist_info):
        """Entropy.

        Args:
            dist_info (dict): Distribution parameters.

        """

    def log_likelihood_sym(self, x_var, dist_info_vars):
        """Log Likelihood sym.

        Args:
            x_var (tf.Tensor): Input placeholder.
            dist_info_vars (dict): Distribution tensors.

        """

    def log_likelihood(self, xs, dist_info):
        """Log Likelihood.

        Args:
            xs (np.ndarray): Inputs.
            dist_info (dict): Distribution parameters.

        """

    @property
    def dist_info_specs(self):
        """Distribution information specification.

        Returns:
            list[tuple]: Key for distribution information and shape.

        """
        return [('dummy', (1, ))]
