"""A baseline based on a GaussianCNN model."""
import akro
from dowel import tabular
import numpy as np
import tensorflow as tf

from garage import make_optimizer
from garage.misc.tensor_utils import normalize_pixel_batch
from garage.np.baselines.baseline import Baseline
from garage.tf.baselines.gaussian_cnn_regressor_model import (
    GaussianCNNRegressorModel)
from garage.tf.misc import tensor_utils
from garage.tf.optimizers import LbfgsOptimizer, PenaltyLbfgsOptimizer


# pylint: disable=too-many-ancestors
class GaussianCNNBaseline(GaussianCNNRegressorModel, Baseline):
    """Fits a Gaussian distribution to the outputs of a CNN.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
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
                 env_spec,
                 filters,
                 strides,
                 padding,
                 hidden_sizes,
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.initializers.glorot_uniform(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(),
                 output_b_init=tf.zeros_initializer(),
                 name='GaussianCNNBaseline',
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

        if not isinstance(env_spec.observation_space, akro.Box) or \
                not len(env_spec.observation_space.shape) in (2, 3):
            raise ValueError(
                '{} can only process 2D, 3D akro.Image or'
                ' akro.Box observations, but received an env_spec with '
                'observation_space of type {} and shape {}'.format(
                    type(self).__name__,
                    type(env_spec.observation_space).__name__,
                    env_spec.observation_space.shape))

        self._env_spec = env_spec
        self._use_trust_region = use_trust_region
        self._subsample_factor = subsample_factor
        self._max_kl_step = max_kl_step
        self._normalize_inputs = normalize_inputs
        self._normalize_outputs = normalize_outputs

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

        super().__init__(input_shape=env_spec.observation_space.shape,
                         output_dim=1,
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
                         layer_normalization=layer_normalization,
                         name=name)
        self._x_mean = None
        self._x_std = None
        self._y_mean = None
        self._y_std = None

        self._initialize()

    def _initialize(self):
        input_var = tf.compat.v1.placeholder(tf.float32,
                                             shape=(None, ) +
                                             self._input_shape)

        ys_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                          name='ys',
                                          shape=(None, self._output_dim))
        old_means_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                                 name='old_means',
                                                 shape=(None,
                                                        self._output_dim))
        old_log_stds_var = tf.compat.v1.placeholder(dtype=tf.float32,
                                                    name='old_log_stds',
                                                    shape=(None,
                                                           self._output_dim))

        (_, means, log_stds, _, norm_means, norm_log_stds, self._x_mean,
         self._x_std, self._y_mean, self._y_std,
         dist) = self.build(input_var).outputs

        normalized_ys_var = (ys_var - self._y_mean) / self._y_std

        normalized_old_means_var = (old_means_var - self._y_mean) / self._y_std
        normalized_old_log_stds_var = (old_log_stds_var -
                                       tf.math.log(self._y_std))

        normalized_dist_info_vars = dict(mean=norm_means,
                                         log_std=norm_log_stds)

        mean_kl = tf.reduce_mean(
            dist.kl_sym(
                dict(mean=normalized_old_means_var,
                     log_std=normalized_old_log_stds_var),
                normalized_dist_info_vars,
            ))

        loss = -tf.reduce_mean(
            dist.log_likelihood_sym(normalized_ys_var,
                                    normalized_dist_info_vars))

        self._f_predict = tensor_utils.compile_function([input_var], means)
        self._f_pdists = tensor_utils.compile_function([input_var],
                                                       [means, log_stds])

        optimizer_args = dict(
            loss=loss,
            target=self,
            network_outputs=[norm_means, norm_log_stds],
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

    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        """
        xs = np.concatenate([p['observations'] for p in paths])
        if isinstance(self.env_spec.observation_space, akro.Image):
            xs = normalize_pixel_batch(xs)

        ys = np.concatenate([p['returns'] for p in paths])
        ys = ys.reshape((-1, 1))

        if self._subsample_factor < 1:
            num_samples_tot = xs.shape[0]
            idx = np.random.randint(
                0, num_samples_tot,
                int(num_samples_tot * self._subsample_factor))
            xs, ys = xs[idx], ys[idx]

        if self._normalize_inputs:
            # recompute normalizing constants for inputs
            self._x_mean.load(np.mean(xs, axis=0, keepdims=True))
            self._x_std.load(np.std(xs, axis=0, keepdims=True) + 1e-8)
        if self._normalize_outputs:
            # recompute normalizing constants for outputs
            self._y_mean.load(np.mean(ys, axis=0, keepdims=True))
            self._y_std.load(np.std(ys, axis=0, keepdims=True) + 1e-8)
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

    def predict(self, paths):
        """Predict ys based on input xs.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        Return:
            numpy.ndarray: The predicted ys.

        """
        xs = paths['observations']
        if isinstance(self.env_spec.observation_space, akro.Image):
            xs = normalize_pixel_batch(xs)

        return self._f_predict(xs).flatten()

    @property
    def recurrent(self):
        """bool: If this module has a hidden state."""
        return False

    @property
    def env_spec(self):
        """Policy environment specification.

        Returns:
            garage.EnvSpec: Environment specification.

        """
        return self._env_spec

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_f_predict']
        del new_dict['_f_pdists']
        del new_dict['_x_mean']
        del new_dict['_x_std']
        del new_dict['_y_mean']
        del new_dict['_y_std']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
