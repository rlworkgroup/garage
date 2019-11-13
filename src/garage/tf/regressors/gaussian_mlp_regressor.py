"""A regressor based on a GaussianMLP model."""
from dowel import tabular
import numpy as np
import tensorflow as tf

from garage.tf.misc import tensor_utils
from garage.tf.optimizers import LbfgsOptimizer, PenaltyLbfgsOptimizer
from garage.tf.regressors import StochasticRegressor
from garage.tf.regressors.gaussian_mlp_regressor_model import (
    GaussianMLPRegressorModel)


class GaussianMLPRegressor(StochasticRegressor):
    """Fits data to a Gaussian whose parameters are estimated by an MLP.

    Args:
        input_shape (tuple[int]): Input shape of the training data.
        output_dim (int): Output dimension of the model.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (Callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (Callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (Callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (Callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (Callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (Callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        optimizer (garage.tf.Optimizer): Optimizer for minimizing the negative
            log-likelihood.
        optimizer_args (dict): Arguments for the optimizer. Default is None,
            which means no arguments.
        use_trust_region (bool): Whether to use trust region constraint.
        max_kl_step (float): KL divergence constraint for each iteration.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
        adaptive_std (bool): Is std a neural network. If False, it will be a
            parameter.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for std. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        std_nonlinearity (Callable): Nonlinearity for each hidden layer in
            the std network.
        layer_normalization (bool): Bool for using layer normalization or not.
        normalize_inputs (bool): Bool for normalizing inputs or not.
        normalize_outputs (bool): Bool for normalizing outputs or not.
        subsample_factor (float): The factor to subsample the data. By default
            it is 1.0, which means using all the data.

    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='GaussianMLPRegressor',
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
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

        with tf.compat.v1.variable_scope(self._name, reuse=False) as vs:
            self._variable_scope = vs
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
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
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
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, ) +
                                             self._input_shape)

        with tf.compat.v1.variable_scope(self._variable_scope):
            self.model.build(input_var)
            ys_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                              name='ys',
                                              shape=(None, self._output_dim))
            old_means_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                                     name='old_means',
                                                     shape=(None,
                                                            self._output_dim))
            old_log_stds_var = tf.compat.v1.placeholder(
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
            normalized_old_log_stds_var = (old_log_stds_var -
                                           tf.math.log(y_std_var))

            normalized_dist_info_vars = dict(mean=normalized_means_var,
                                             log_std=normalized_log_stds_var)

            mean_kl = tf.reduce_mean(
                self.model.networks['default'].dist.kl_sym(
                    dict(mean=normalized_old_means_var,
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
        """Fit with input data xs and label ys.

        Args:
            xs (numpy.ndarray): Input data.
            ys (numpy.ndarray): Label of input data.

        """
        if self._subsample_factor < 1:
            num_samples_tot = xs.shape[0]
            idx = np.random.randint(
                0, num_samples_tot,
                int(num_samples_tot * self._subsample_factor))
            xs, ys = xs[idx], ys[idx]

        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self.model.networks['default'].x_mean.load(
                np.mean(xs, axis=0, keepdims=True))
            self.model.networks['default'].x_std.load(
                np.std(xs, axis=0, keepdims=True) + 1e-8)
        if self._normalize_outputs:
            # recompute normalizing constants for outputs
            self.model.networks['default'].y_mean.load(
                np.mean(ys, axis=0, keepdims=True))
            self.model.networks['default'].y_std.load(
                np.std(ys, axis=0, keepdims=True) + 1e-8)
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
        """Predict ys based on input xs.

        Args:
            xs (numpy.ndarray): Input data.

        Return:
            np.ndarray: The predicted ys.

        """
        return self._f_predict(xs)

    def log_likelihood_sym(self, x_var, y_var, name=None):
        """Create a symbolic graph of the log likelihood.

        Args:
            x_var (tf.Tensor): Input tf.Tensor for the input data.
            y_var (tf.Tensor): Input tf.Tensor for the label of data.
            name (str): Name of the new graph.

        Return:
            tf.Tensor: Output of the symbolic log-likelihood graph.

        """
        params = self.dist_info_sym(x_var, name=name)
        means_var = params['mean']
        log_stds_var = params['log_std']

        return self.model.networks[name].dist.log_likelihood_sym(
            y_var, dict(mean=means_var, log_std=log_stds_var))

    def dist_info_sym(self, x_var, name=None):
        """Create a symbolic graph of the distribution parameters.

        Args:
            x_var (tf.Tensor): tf.Tensor of the input data.
            name (str): Name of the new graph.

        Return:
            dict[tf.Tensor]: Outputs of the symbolic distribution parameter
                graph.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            self.model.build(x_var, name=name)

        means_var = self.model.networks[name].means
        log_stds_var = self.model.networks[name].log_stds

        return dict(mean=means_var, log_std=log_stds_var)

    def get_params_internal(self, **args):
        """Get the params, which are the trainable variables."""
        del args
        return self._variable_scope.trainable_variables()

    def __getstate__(self):
        """See `Object.__getstate__`."""
        new_dict = super().__getstate__()
        del new_dict['_f_predict']
        del new_dict['_f_pdists']
        return new_dict

    def __setstate__(self, state):
        """See `Object.__setstate__`."""
        super().__setstate__(state)
        self._initialize()
