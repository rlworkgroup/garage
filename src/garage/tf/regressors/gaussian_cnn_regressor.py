"""A regressor based on a GaussianMLP model."""
from dowel import tabular
import numpy as np
import tensorflow as tf

from garage import make_optimizer
from garage.experiment import deterministic
from garage.tf.misc import tensor_utils
from garage.tf.optimizers import LbfgsOptimizer, PenaltyLbfgsOptimizer
from garage.tf.regressors.gaussian_cnn_regressor_model import (
    GaussianCNNRegressorModel)
from garage.tf.regressors.regressor import StochasticRegressor


class GaussianCNNRegressor(StochasticRegressor):
    """Fits a Gaussian distribution to the outputs of a CNN.

    Args:
        input_shape(tuple[int]): Input shape of the model (without the batch
            dimension).
        output_dim (int): Output dimension of the model.
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
        strides(tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the Convolutional model for mean. For example, (32, 32) means the
            network consists of two dense layers, each with 32 hidden units.
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
        name (str): Name of this model (also used as its scope).
        learn_std (bool): Whether to train the standard deviation parameter of
            the Gaussian distribution.
        init_std (float): Initial standard deviation for the Gaussian
            distribution.
        adaptive_std (bool): Whether to use a neural network to learn the
            standard deviation of the Gaussian distribution. Unless True, the
            standard deviation is learned as a parameter which is not
            conditioned on the inputs.
        std_share_network (bool): Boolean for whether the mean and standard
            deviation models share a CNN network. If True, each is a head from
            a single body network. Otherwise, the parameters are estimated
            using the outputs of two indepedent networks.
        std_filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and
            dimension of filters. For example, ((3, (3, 5)), (32, (3, 3)))
            means there are two convolutional layers. The filter for the first
            layer have 3 channels and its shape is (3 x 5), while the filter
            for the second layer have 32 channels and its shape is (3 x 3).
        std_strides(tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        std_padding (str): The type of padding algorithm to use in std network,
            either 'SAME' or 'VALID'.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the Conv for std. For example, (32, 32) means the Conv consists
            of two hidden layers, each with 32 hidden units.
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_output_nonlinearity (Callable): Activation function for output
            dense layer in the std network. It should return a tf.Tensor. Set
            it to None to maintain a linear activation.
        layer_normalization (bool): Bool for using layer normalization or not.
        normalize_inputs (bool): Bool for normalizing inputs or not.
        normalize_outputs (bool): Bool for normalizing outputs or not.
        subsample_factor (float): The factor to subsample the data. By default
            it is 1.0, which means using all the data.
        optimizer (garage.tf.Optimizer): Optimizer used for fitting the model.
        optimizer_args (dict): Arguments for the optimizer. Default is None,
            which means no arguments.
        use_trust_region (bool): Whether to use a KL-divergence constraint.
        max_kl_step (float): KL divergence constraint for each iteration, if
            `use_trust_region` is active.

    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 filters,
                 strides,
                 padding,
                 hidden_sizes,
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 name='GaussianCNNRegressor',
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_filters=(),
                 std_strides=(),
                 std_padding='SAME',
                 std_hidden_sizes=(),
                 std_hidden_nonlinearity=None,
                 std_output_nonlinearity=None,
                 layer_normalization=False,
                 normalize_inputs=True,
                 normalize_outputs=True,
                 subsample_factor=1.,
                 optimizer=None,
                 optimizer_args=None,
                 use_trust_region=True,
                 max_kl_step=0.01):

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
                    self._optimizer = make_optimizer(PenaltyLbfgsOptimizer,
                                                     **optimizer_args)
                else:
                    self._optimizer = make_optimizer(LbfgsOptimizer,
                                                     **optimizer_args)
            else:
                self._optimizer = make_optimizer(optimizer, **optimizer_args)

        self.model = GaussianCNNRegressorModel(
            input_shape=input_shape,
            output_dim=output_dim,
            filters=filters,
            strides=strides,
            padding=padding,
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
            std_filters=std_filters,
            std_strides=std_strides,
            std_padding=std_padding,
            std_hidden_sizes=std_hidden_sizes,
            std_hidden_nonlinearity=std_hidden_nonlinearity,
            std_output_nonlinearity=std_output_nonlinearity,
            std_parameterization='exp',
            layer_normalization=layer_normalization)
        self._network = None

        self._initialize()

    def _initialize(self):
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, ) +
                                             self._input_shape)

        with tf.compat.v1.variable_scope(self._variable_scope):
            self._network = self.model.build(input_var)
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

            y_mean_var = self._network.y_mean
            y_std_var = self._network.y_std
            means_var = self._network.means
            log_stds_var = self._network.log_stds
            normalized_means_var = self._network.normalized_means
            normalized_log_stds_var = self._network.normalized_log_stds

            normalized_ys_var = (ys_var - y_mean_var) / y_std_var

            normalized_old_means_var = (old_means_var - y_mean_var) / y_std_var
            normalized_old_log_stds_var = (old_log_stds_var -
                                           tf.math.log(y_std_var))

            normalized_dist_info_vars = dict(mean=normalized_means_var,
                                             log_std=normalized_log_stds_var)

            mean_kl = tf.reduce_mean(
                self._network.dist.kl_sym(
                    dict(mean=normalized_old_means_var,
                         log_std=normalized_old_log_stds_var),
                    normalized_dist_info_vars,
                ))

            loss = -tf.reduce_mean(
                self._network.dist.log_likelihood_sym(
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
            self._network.x_mean.load(np.mean(xs, axis=0, keepdims=True))
            self._network.x_std.load(np.std(xs, axis=0, keepdims=True) + 1e-8)
        if self._normalize_outputs:
            # recompute normalizing constants for outputs
            self._network.y_mean.load(np.mean(ys, axis=0, keepdims=True))
            self._network.y_std.load(np.std(ys, axis=0, keepdims=True) + 1e-8)
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
            numpy.ndarray: The predicted ys.

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

        return self._network.dist.log_likelihood_sym(
            y_var, dict(mean=means_var, log_std=log_stds_var))

    def dist_info_sym(self, input_var, state_info_vars=None, name=None):
        """Create a symbolic graph of the distribution parameters.

        Args:
            input_var (tf.Tensor): tf.Tensor of the input data.
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the regressor at the time it
                received the input.
            name (str): Name of the new graph.

        Return:
            dict[tf.Tensor]: Outputs of the symbolic distribution parameter
                graph.

        """
        del state_info_vars
        with tf.compat.v1.variable_scope(self._variable_scope):
            network = self.model.build(input_var, name=name)

        means_var = network.means
        log_stds_var = network.log_stds

        return dict(mean=means_var, log_std=log_stds_var)

    @property
    def recurrent(self):
        """bool: If this module has a hidden state."""
        return False

    @property
    def vectorized(self):
        """bool: If this module supports vectorization input."""
        return True

    @property
    def distribution(self):
        """garage.tf.distributions.DiagonalGaussian: Distribution."""
        return self._network.dist

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_predict']
        del new_dict['_f_pdists']
        del new_dict['_network']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
