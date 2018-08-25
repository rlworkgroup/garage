import theano.tensor as TT


class Distribution:
    @property
    def dim(self):
        raise NotImplementedError

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two distributions
        """
        raise NotImplementedError

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two distributions
        """
        raise NotImplementedError

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars,
                             new_dist_info_vars):
        raise NotImplementedError

    def entropy(self, dist_info):
        raise NotImplementedError

    def log_likelihood_sym(self, x_var, dist_info_vars):
        raise NotImplementedError

    def likelihood_sym(self, x_var, dist_info_vars):
        return TT.exp(self.log_likelihood_sym(x_var, dist_info_vars))

    def log_likelihood(self, xs, dist_info):
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        raise NotImplementedError
