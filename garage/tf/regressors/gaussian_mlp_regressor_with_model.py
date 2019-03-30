"""A regressor based on a GaussianMLP model."""
import numpy as np
import tensorflow as tf

from garage.logger import tabular
from garage.tf.misc import tensor_utils
from garage.tf.optimizers import LbfgsOptimizer, PenaltyLbfgsOptimizer
from garage.tf.regressors import GaussianMLPRegressorModel
from garage.tf.regressors import StochasticRegressor


class GaussianMLPRegressorWithModel(StochasticRegressor):
    """
    GaussianMLPRegressorWithModel.

    A class for performing regression by fitting a Gaussian distribution
    to the outputs.

    :param input_shape: Shape of the input data.
    :param output_dim: Dimension of output.
    :param hidden_sizes: Number of hidden units of each layer of the mean
     network.
    :param hidden_nonlinearity: Non-linearity used for each layer of the
     mean network.
    :param optimizer: Optimizer for minimizing the negative log-likelihood.
    :param use_trust_region: Whether to use trust region constraint.
    :param max_kl_step: KL divergence constraint for each iteration
    :param learn_std: Whether to learn the standard deviations. Only
     effective if adaptive_std is False. If adaptive_std is True, this
     parameter is ignored, and the weights for the std network are always
     earned.
    :param adaptive_std: Whether to make the std a function of the states.
    :param std_share_network: Whether to use the same network as the mean.
    :param std_hidden_sizes: Number of hidden units of each layer of the
     std network. Only used if `std_share_network` is False. It defaults to
     the same architecture as the mean.
    :param std_nonlinearity: Non-linearity used for each layer of the std
     network. Only used if `std_share_network` is False. It defaults to the
     same non-linearity as the mean.
    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='GaussianMLPRegressorWithModel',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 optimizer=None,
                 optimizer_args=None,
                 use_trust_region=True,
                 max_kl_step=0.01,
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_hidden_sizes=(32, 32),
                 std_nonlinearity=None,
                 layer_normalization=False,
                 normalize_inputs=True,
                 normalize_outputs=True,
                 subsample_factor=1.0):
        super().__init__(input_shape, output_dim, name)
        self._use_trust_region = use_trust_region
        self._subsample_factor = subsample_factor
        self._max_kl_step = max_kl_step
        self._normalize_inputs = normalize_inputs
        self._normalize_outputs = normalize_outputs

        with tf.variable_scope(
                self._name, reuse=False) as self._variable_scope:
            if optimizer_args is None:
                optimizer_args = dict()
            if optimizer is None:
                if use_trust_region:
                    optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
                else:
                    optimizer = LbfgsOptimizer(**optimizer_args)
            else:
                optimizer = optimizer(**optimizer_args)
            self._optimizer = optimizer

        self.model = GaussianMLPRegressorModel(
            input_shape=input_shape,
            output_dim=self._output_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            output_nonlinearity=None,
            learn_std=learn_std,
            adaptive_std=adaptive_std,
            std_share_network=std_share_network,
            init_std=init_std,
            min_std=None,
            max_std=None,
            std_hidden_sizes=std_hidden_sizes,
            std_hidden_nonlinearity=std_nonlinearity,
            std_output_nonlinearity=None,
            std_parameterization='exp',
            layer_normalization=layer_normalization)

        self._initialize()

    def _initialize(self):
        input_var = tf.placeholder(
            tf.float32, shape=(None, ) + self._input_shape)

        with tf.variable_scope(self._variable_scope):
            self.model.build(input_var)
            ys_var = tf.placeholder(
                dtype=tf.float32, name='ys', shape=(None, self._output_dim))
            old_means_var = tf.placeholder(
                dtype=tf.float32, name='ys', shape=(None, self._output_dim))
            old_log_stds_var = tf.placeholder(
                dtype=tf.float32,
                name='old_log_stds',
                shape=(None, self._output_dim))

            y_mean_var = self.model.networks['default'].y_mean
            y_std_var = self.model.networks['default'].y_std
            means_var = self.model.networks['default'].means
            log_stds_var = self.model.networks['default'].log_stds
            normalized_means_var = self.model.networks[
                'default'].normalized_means
            normalized_log_stds_var = self.model.networks[
                'default'].normalized_log_stds

            normalized_ys_var = (ys_var - y_mean_var) / y_std_var

            normalized_old_means_var = (old_means_var - y_mean_var) / y_std_var
            normalized_old_log_stds_var = old_log_stds_var - tf.log(y_std_var)

            normalized_dist_info_vars = dict(
                mean=normalized_means_var, log_std=normalized_log_stds_var)

            mean_kl = tf.reduce_mean(
                self.model.networks['default'].dist.kl_sym(
                    dict(
                        mean=normalized_old_means_var,
                        log_std=normalized_old_log_stds_var),
                    normalized_dist_info_vars,
                ))

            loss = -tf.reduce_mean(
                self.model.networks['default'].dist.log_likelihood_sym(
                    normalized_ys_var, normalized_dist_info_vars))

            self._f_predict = tensor_utils.compile_function([input_var],
                                                            means_var)
            self._f_pdists = tensor_utils.compile_function(
                [input_var], [means_var, log_stds_var])

            optimizer_args = dict(
                loss=loss,
                target=self,
                network_outputs=[
                    normalized_means_var, normalized_log_stds_var
                ],
            )

            if self._use_trust_region:
                optimizer_args['leq_constraint'] = (mean_kl, self._max_kl_step)
                optimizer_args['inputs'] = [
                    input_var, ys_var, old_means_var, old_log_stds_var
                ]
            else:
                optimizer_args['inputs'] = [input_var, ys_var]

            with tf.name_scope('update_opt'):
                self._optimizer.update_opt(**optimizer_args)

    def fit(self, xs, ys):
        """Fit with input data xs and label ys."""
        if self._subsample_factor < 1:
            num_samples_tot = xs.shape[0]
            idx = np.random.randint(
                0, num_samples_tot,
                int(num_samples_tot * self._subsample_factor))
            xs, ys = xs[idx], ys[idx]

        sess = tf.get_default_session()
        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            sess.run([
                tf.assign(self.model.networks['default'].x_mean,
                          np.mean(xs, axis=0, keepdims=True)),
                tf.assign(self.model.networks['default'].x_std,
                          np.std(xs, axis=0, keepdims=True) + 1e-8),
            ])
        if self._normalize_outputs:
            # recompute normalizing constants for outputs
            sess.run([
                tf.assign(self.model.networks['default'].y_mean,
                          np.mean(ys, axis=0, keepdims=True)),
                tf.assign(self.model.networks['default'].y_std,
                          np.std(ys, axis=0, keepdims=True) + 1e-8),
            ])
        if self._use_trust_region:
            old_means, old_log_stds = self._f_pdists(xs)
            inputs = [xs, ys, old_means, old_log_stds]
        else:
            inputs = [xs, ys]
        loss_before = self._optimizer.loss(inputs)
        tabular.record('{}/LossBefore'.format(self._name), loss_before)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        tabular.record('{}/LossAfter'.format(self._name), loss_after)
        if self._use_trust_region:
            tabular.record('{}/MeanKL'.format(self._name),
                           self._optimizer.constraint_val(inputs))
        tabular.record('{}/dLoss'.format(self._name), loss_before - loss_after)

    def predict(self, xs):
        """
        Return the maximum likelihood estimate of the predicted y.

        :param xs:
        :return:

        """
        return self._f_predict(xs)

    def sample_predict(self, xs):
        """
        Sample one possible output from the prediction distribution.

        :param xs:
        :return:

        """
        means, log_stds = self._f_pdists(xs)
        return self.model.networks['default'].dist.sample(
            dict(mean=means, log_std=log_stds))

    def predict_log_likelihood(self, xs, ys):
        """
        Return the maximum likelihood estimate of the predicted y and input ys.

        Args:
            xs: Input data.
            ys: Label of input data.
        """
        means, log_stds = self._f_pdists(xs)
        return self.model.networks['default'].dist.log_likelihood(
            ys, dict(mean=means, log_std=log_stds))

    def log_likelihood_sym(self, x_var, y_var, name=None):
        """
        Symbolic graph of the log likelihood.

        Args:
            x_var: Input tf.Tensor for the input data.
            y_var: Input tf.Tensor for the label of data.
            name: Name of the new graph.
        """
        with tf.variable_scope(self._variable_scope):
            self.model.build(x_var, name=name)

        means_var = self.model.networks[name].mean
        log_stds_var = self.model.networks[name].log_std

        return self.model.networks[name].dist.log_likelihood_sym(
            y_var, dict(mean=means_var, log_std=log_stds_var))
