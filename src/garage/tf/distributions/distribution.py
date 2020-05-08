"""Distributions Base."""


class Distribution:
    """Base class for distribution."""

    @property
    def dim(self):
        """int: Dimension of this distribution."""
        raise NotImplementedError

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars, name='kl_sym'):
        """Compute the symbolic KL divergence of two distributions.

        Args:
            old_dist_info_vars (tf.Tensor): Symbolic parameters of
                the old distribution.
            new_dist_info_vars (tf.Tensor): Symbolic parameters of
                the new distribution.
            name (str): TensorFlow scope name.

        Returns:
            tf.Tensor: Symbolic KL divergence between the two distributions.

        """
        raise NotImplementedError

    def kl(self, old_dist_info, new_dist_info):
        """Compute the KL divergence of two distributions.

        Args:
            old_dist_info (dict): Parameters of the old distribution.
            new_dist_info (dict): Parameters of the new distribution.

        Returns:
            float: KL Divergence between two distributions.

        """
        raise NotImplementedError

    def likelihood_ratio_sym(self,
                             x_var,
                             old_dist_info_vars,
                             new_dist_info_vars,
                             name='ll_ratio_sym'):
        """Symbolic likelihood ratio.

        Args:
            x_var (tf.Tensor): Input placeholder.
            old_dist_info_vars (dict): Old distribution tensors.
            new_dist_info_vars (dict): New distribution tensors.
            name (str): TensorFlow scope name.

        Returns:
            tf.Tensor: Symbolic likelihood ratio.

        """
        raise NotImplementedError

    def entropy(self, dist_info):
        """Entropy of a distribution.

        Args:
            dist_info (dict): Parameters of a distribution.

        Returns:
            float: Entropy of the distribution.

        """
        raise NotImplementedError

    def entropy_sym(self, dist_info_vars, name='entropy_sym'):
        """Symbolic entropy of a distribution.

        Args:
            dist_info_vars (dict): Symbolic parameters of a distribution.
            name (str): TensorFlow scope name.

        Returns:
            tf.Tensor: Symbolic entropy of the distribution.

        """
        raise NotImplementedError

    def log_likelihood_sym(self, x_var, dist_info_vars, name='ll_sym'):
        """Symbolic log likelihood.

        Args:
            x_var (tf.Tensor): Input placeholder.
            dist_info_vars (dict): Parameters of a distribution.
            name (str): TensorFlow scope name.

        Returns:
            tf.Tensor: Symbolic log likelihood.

        """
        raise NotImplementedError

    def log_likelihood(self, xs, dist_info):
        """Log likelihood of a sample under a distribution.

        Args:
            xs (np.ndarray): Input value.
            dist_info (dict): Parameters of a distribution.

        Returns:
            float: Log likelihood of a sample under the distribution.

        """
        raise NotImplementedError

    @property
    def dist_info_specs(self):
        """list: Specification of the parameter of a distribution."""
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        """list: Parameter names."""
        return [k for k, _ in self.dist_info_specs]
