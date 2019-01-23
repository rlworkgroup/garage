"""Gaussian Conv Regressor."""
import numpy as np
import tensorflow as tf

from garage.core import Serializable
from garage.misc import logger
from garage.tf.core import LayersPowered
from garage.tf.core import Parameterized
import garage.tf.core.layers as layers
from garage.tf.core.network import ConvNetwork
from garage.tf.distributions import DiagonalGaussian
from garage.tf.misc import tensor_utils
from garage.tf.optimizers import LbfgsOptimizer
from garage.tf.optimizers import PenaltyLbfgsOptimizer


class GaussianConvRegressor(LayersPowered, Serializable, Parameterized):
    """
    A regressor fits a Gaussian distribution to outputs using a ConvNetwork.

    Args:
        Most of the net param infos can be found at the docstring of
         garage.tf.core.network.ConvNetwork.
        input_shape: a tuple or list contains input shapes of the mean net.
        conv_filters: a list of numbers of the mean convnet kernels.
        conv_filter_sizes: a list of sizes of the mean convnet kernels.
        conv_strides: a list of strides of the mean convnet kernels.
        conv_pads: a list of pad formats of the mean convnet.
        hidden_sizes: a list of numbers of hidden units for all fc layers.
        hidden_nonlinearity: a nonlinearity from tf.nn, shared by all conv and
         fc layers.

    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 conv_filters,
                 conv_filter_sizes,
                 conv_strides,
                 conv_pads,
                 hidden_sizes,
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 name="GaussianConvRegressor",
                 mean_network=None,
                 learn_std=True,
                 init_std=1.0,
                 adaptive_std=False,
                 std_share_network=False,
                 std_conv_filters=[],
                 std_conv_filter_sizes=[],
                 std_conv_strides=[],
                 std_conv_pads=[],
                 std_hidden_sizes=[],
                 std_hidden_nonlinearity=None,
                 std_output_nonlinearity=None,
                 normalize_inputs=True,
                 normalize_outputs=True,
                 subsample_factor=1.,
                 optimizer=None,
                 optimizer_args=dict(),
                 use_trust_region=True,
                 step_size=0.01):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())
        self._mean_network_name = "mean_network"
        self._std_network_name = "std_network"

        with tf.variable_scope(name):
            if optimizer is None:
                if use_trust_region:
                    optimizer = PenaltyLbfgsOptimizer(**optimizer_args)
                else:
                    optimizer = LbfgsOptimizer(**optimizer_args)
            else:
                optimizer = optimizer(**optimizer_args)

            self._optimizer = optimizer
            self._subsample_factor = subsample_factor

            if mean_network is None:
                if std_share_network:
                    b = np.concatenate(
                        [
                            np.zeros(output_dim),
                            np.full(output_dim, np.log(init_std))
                        ],
                        axis=0)  # yapf: disable
                    b = tf.constant_initializer(b)
                    mean_network = ConvNetwork(
                        name=self._mean_network_name,
                        input_shape=input_shape,
                        output_dim=2 * output_dim,
                        conv_filters=conv_filters,
                        conv_filter_sizes=conv_filter_sizes,
                        conv_strides=conv_strides,
                        conv_pads=conv_pads,
                        hidden_sizes=hidden_sizes,
                        hidden_nonlinearity=hidden_nonlinearity,
                        output_nonlinearity=output_nonlinearity,
                        output_b_init=b)
                    l_mean = layers.SliceLayer(
                        mean_network.output_layer,
                        slice(output_dim),
                        name="mean_slice",
                    )
                else:
                    mean_network = ConvNetwork(
                        name=self._mean_network_name,
                        input_shape=input_shape,
                        output_dim=output_dim,
                        conv_filters=conv_filters,
                        conv_filter_sizes=conv_filter_sizes,
                        conv_strides=conv_strides,
                        conv_pads=conv_pads,
                        hidden_sizes=hidden_sizes,
                        hidden_nonlinearity=hidden_nonlinearity,
                        output_nonlinearity=output_nonlinearity)
                    l_mean = mean_network.output_layer

            if adaptive_std:
                l_log_std = ConvNetwork(
                    name=self._std_network_name,
                    input_shape=input_shape,
                    output_dim=output_dim,
                    conv_filters=std_conv_filters,
                    conv_filter_sizes=std_conv_filter_sizes,
                    conv_strides=std_conv_strides,
                    conv_pads=std_conv_pads,
                    hidden_sizes=std_hidden_sizes,
                    hidden_nonlinearity=std_hidden_nonlinearity,
                    output_nonlinearity=std_output_nonlinearity,
                    output_b_init=tf.constant_initializer(np.log(init_std)),
                ).output_layer
            elif std_share_network:
                l_log_std = layers.SliceLayer(
                    mean_network.output_layer,
                    slice(output_dim, 2 * output_dim),
                    name="log_std_slice",
                )
            else:
                l_log_std = layers.ParamLayer(
                    mean_network.input_layer,
                    num_units=output_dim,
                    param=tf.constant_initializer(np.log(init_std)),
                    trainable=learn_std,
                    name=self._std_network_name,
                )

            LayersPowered.__init__(self, [l_mean, l_log_std])

            xs_var = mean_network.input_layer.input_var
            ys_var = tf.placeholder(
                dtype=tf.float32, name="ys", shape=(None, output_dim))
            old_means_var = tf.placeholder(
                dtype=tf.float32, name="ys", shape=(None, output_dim))
            old_log_stds_var = tf.placeholder(
                dtype=tf.float32,
                name="old_log_stds",
                shape=(None, output_dim))

            x_mean_var = tf.Variable(
                np.zeros((1, np.prod(input_shape)), dtype=np.float32),
                name="x_mean",
            )
            x_std_var = tf.Variable(
                np.ones((1, np.prod(input_shape)), dtype=np.float32),
                name="x_std",
            )
            y_mean_var = tf.Variable(
                np.zeros((1, output_dim), dtype=np.float32),
                name="y_mean",
            )
            y_std_var = tf.Variable(
                np.ones((1, output_dim), dtype=np.float32),
                name="y_std",
            )

            normalized_xs_var = (xs_var - x_mean_var) / x_std_var
            normalized_ys_var = (ys_var - y_mean_var) / y_std_var

            with tf.name_scope(
                    self._mean_network_name, values=[normalized_xs_var]):
                normalized_means_var = layers.get_output(
                    l_mean, {mean_network.input_layer: normalized_xs_var})
            with tf.name_scope(
                    self._std_network_name, values=[normalized_xs_var]):
                normalized_log_stds_var = layers.get_output(
                    l_log_std, {mean_network.input_layer: normalized_xs_var})

            means_var = normalized_means_var * y_std_var + y_mean_var
            log_stds_var = normalized_log_stds_var + tf.log(y_std_var)

            normalized_old_means_var = (old_means_var - y_mean_var) / y_std_var
            normalized_old_log_stds_var = old_log_stds_var - tf.log(y_std_var)

            dist = self._dist = DiagonalGaussian(output_dim)

            normalized_dist_info_vars = dict(
                mean=normalized_means_var, log_std=normalized_log_stds_var)

            mean_kl = tf.reduce_mean(
                dist.kl_sym(
                    dict(
                        mean=normalized_old_means_var,
                        log_std=normalized_old_log_stds_var),
                    normalized_dist_info_vars,
                ))

            loss = -tf.reduce_mean(
                dist.log_likelihood_sym(normalized_ys_var,
                                        normalized_dist_info_vars))

            self._f_predict = tensor_utils.compile_function([xs_var],
                                                            means_var)
            self._f_pdists = tensor_utils.compile_function(
                [xs_var], [means_var, log_stds_var])
            self._l_mean = l_mean
            self._l_log_std = l_log_std

            optimizer_args = dict(
                loss=loss,
                target=self,
                network_outputs=[
                    normalized_means_var, normalized_log_stds_var
                ],
            )

            if use_trust_region:
                optimizer_args["leq_constraint"] = (mean_kl, step_size)
                optimizer_args["inputs"] = [
                    xs_var, ys_var, old_means_var, old_log_stds_var
                ]
            else:
                optimizer_args["inputs"] = [xs_var, ys_var]

            self._optimizer.update_opt(**optimizer_args)

            self._use_trust_region = use_trust_region
            self._name = name

            self._normalize_inputs = normalize_inputs
            self._normalize_outputs = normalize_outputs
            self._mean_network = mean_network
            self._x_mean_var = x_mean_var
            self._x_std_var = x_std_var
            self._y_mean_var = y_mean_var
            self._y_std_var = y_std_var

    def fit(self, xs, ys):
        """Optimize the regressor based on the inputs."""
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
                tf.assign(self._x_mean_var, np.mean(xs, axis=0,
                                                    keepdims=True)),
                tf.assign(self._x_std_var,
                          np.std(xs, axis=0, keepdims=True) + 1e-8),
            ])
        if self._normalize_outputs:
            # recompute normalizing constants for outputs
            sess.run([
                tf.assign(self._y_mean_var, np.mean(ys, axis=0,
                                                    keepdims=True)),
                tf.assign(self._y_std_var,
                          np.std(ys, axis=0, keepdims=True) + 1e-8),
            ])
        if self._use_trust_region:
            old_means, old_log_stds = self._f_pdists(xs)
            inputs = [xs, ys, old_means, old_log_stds]
        else:
            inputs = [xs, ys]
        loss_before = self._optimizer.loss(inputs)
        if self._name:
            prefix = self._name + "/"
        else:
            prefix = ""
        logger.record_tabular(prefix + 'LossBefore', loss_before)
        self._optimizer.optimize(inputs)
        loss_after = self._optimizer.loss(inputs)
        logger.record_tabular(prefix + 'LossAfter', loss_after)
        if self._use_trust_region:
            logger.record_tabular(prefix + 'MeanKL',
                                  self._optimizer.constraint_val(inputs))
        logger.record_tabular(prefix + 'dLoss', loss_before - loss_after)

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
        return self._dist.sample(dict(mean=means, log_std=log_stds))

    def predict_log_likelihood(self, xs, ys):
        """
        Predict the log likelihood of the distribution.

        :param xs:
        :param ys:
        :return:

        """
        means, log_stds = self._f_pdists(xs)
        return self._dist.log_likelihood(ys, dict(
            mean=means, log_std=log_stds))

    def log_likelihood_sym(self, x_var, y_var, name=None):
        """
        Return the symbolic log likelihood info of the distribution.

        :param x_var:
        :param y_var:
        :param name:

        """
        with tf.name_scope(name, "log_likelihood_sym", [x_var, y_var]):
            normalized_xs_var = (x_var - self._x_mean_var) / self._x_std_var

            with tf.name_scope(
                    self._mean_network_name, values=[normalized_xs_var]):
                normalized_means_var = layers.get_output(
                    self._l_mean,
                    {self._mean_network.input_layer: normalized_xs_var})
            with tf.name_scope(
                    self._std_network_name, values=[normalized_xs_var]):
                normalized_log_stds_var = layers.get_output(
                    self._l_log_std,
                    {self._mean_network.input_layer: normalized_xs_var})

            means_var = (
                normalized_means_var * self._y_std_var + self._y_mean_var)
            log_stds_var = normalized_log_stds_var + tf.log(self._y_std_var)

            return self._dist.log_likelihood_sym(
                y_var, dict(mean=means_var, log_std=log_stds_var))
