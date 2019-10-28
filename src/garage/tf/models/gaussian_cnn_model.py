"""GaussianCNNModel."""
import numpy as np
import tensorflow as tf

from garage.tf.distributions import DiagonalGaussian
from garage.tf.models.base import Model
from garage.tf.models.cnn import cnn
from garage.tf.models.mlp import mlp
from garage.tf.models.parameter import parameter


class GaussianCNNModel(Model):
    """GaussianCNNModel.

    Args:
        filter_dims(tuple[int]): Dimension of the filters. For example,
            (3, 5) means there are two convolutional layers. The filter
            for first layer is of dimension (3 x 3) and the second one is of
            dimension (5 x 5).
        num_filters(tuple[int]): Number of filters. For example, (3, 32) means
            there are two convolutional layers. The filter for the first layer
            has 3 channels and the second one with 32 channels.
        strides(tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        output_dim (int): Output dimension of the model.
        name (str): Model name, also the variable scope.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the Convolutional model for mean. For example, (32, 32) means the
            network consists of two dense layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
        adaptive_std (bool): Is std a neural network. If False, it will be a
            parameter.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        std_filter_dims(tuple[int]): Dimension of the filters. For example,
            (3, 5) means there are two convolutional layers. The filter
            for first layer is of dimension (3 x 3) and the second one is of
            dimension (5 x 5).
        std_num_filters(tuple[int]): Number of filters. For example, (3, 32)
            means there are two convolutional layers. The filter for the first
            layer has 3 channels and the second one with 32 channels.
        std_strides(tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        std_padding (str): The type of padding algorithm to use in std network,
            either 'SAME' or 'VALID'.
        std_hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the Conv for std. For example, (32, 32) means the Conv consists
            of two hidden layers, each with 32 hidden units.
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues.
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues.
        std_hidden_nonlinearity: Nonlinearity for each hidden layer in
            the std network.
        std_output_nonlinearity (callable): Activation function for output
            dense layer in the std network. It should return a tf.Tensor. Set
            it to None to maintain a linear activation.
        std_output_w_init (callable): Initializer function for the weight
            of output dense layer(s) in the std network.
        std_parameterization (str): How the std should be parametrized. There
            are two options:
            - exp: the logarithm of the std will be stored, and applied a
               exponential transformation
            - softplus: the std will be computed as log(1+exp(x))
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 output_dim,
                 filter_dims,
                 num_filters,
                 strides,
                 padding,
                 hidden_sizes,
                 name=None,
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 learn_std=True,
                 adaptive_std=False,
                 std_share_network=False,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_filter_dims=[],
                 std_num_filters=[],
                 std_strides=[],
                 std_padding='SAME',
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=tf.nn.tanh,
                 std_hidden_w_init=tf.glorot_uniform_initializer(),
                 std_hidden_b_init=tf.zeros_initializer(),
                 std_output_nonlinearity=None,
                 std_output_w_init=tf.glorot_uniform_initializer(),
                 std_parameterization='exp',
                 layer_normalization=False):
        # Network parameters
        super().__init__(name)
        self._output_dim = output_dim
        self._num_filters = num_filters
        self._filter_dims = filter_dims
        self._strides = strides
        self._padding = padding
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._learn_std = learn_std
        self._adaptive_std = adaptive_std
        self._std_share_network = std_share_network
        self._init_std = init_std
        self._min_std = min_std
        self._max_std = max_std
        self._std_num_filters = std_num_filters
        self._std_filter_dims = std_filter_dims
        self._std_strides = std_strides
        self._std_padding = std_padding
        self._std_hidden_sizes = std_hidden_sizes
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_hidden_w_init = std_hidden_w_init
        self._std_hidden_b_init = std_hidden_b_init
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_output_w_init = std_output_w_init
        self._std_parameterization = std_parameterization
        self._layer_normalization = layer_normalization

        # Tranform std arguments to parameterized space
        self._init_std_param = None
        self._min_std_param = None
        self._max_std_param = None
        if self._std_parameterization == 'exp':
            self._init_std_param = np.log(init_std)
            if min_std is not None:
                self._min_std_param = np.log(min_std)
            if max_std is not None:
                self._max_std_param = np.log(max_std)
        elif self._std_parameterization == 'softplus':
            self._init_std_param = np.log(np.exp(init_std) - 1)
            if min_std is not None:
                self._min_std_param = np.log(np.exp(min_std) - 1)
            if max_std is not None:
                self._max_std_param = np.log(np.exp(max_std) - 1)
        else:
            raise NotImplementedError

    def network_output_spec(self):
        """Network output spec."""
        return ['sample', 'mean', 'log_std', 'std_param', 'dist']

    def _build(self, state_input, name=None):
        action_dim = self._output_dim

        with tf.compat.v1.variable_scope('dist_params'):
            if self._std_share_network:
                # mean and std networks share an CNN
                b = np.concatenate([
                    np.zeros(action_dim),
                    np.full(action_dim, self._init_std_param)
                ], axis=0)  # yapf: disable

                mean_std_conv = cnn(
                    input_var=state_input,
                    filter_dims=self._filter_dims,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    hidden_w_init=self._hidden_w_init,
                    hidden_b_init=self._hidden_b_init,
                    num_filters=self._num_filters,
                    strides=self._strides,
                    padding=self._padding,
                    name='mean_std_cnn')
                mean_std_network = mlp(
                    mean_std_conv,
                    output_dim=action_dim * 2,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    hidden_w_init=self._hidden_w_init,
                    hidden_b_init=self._hidden_b_init,
                    output_nonlinearity=self._output_nonlinearity,
                    output_w_init=self._output_w_init,
                    output_b_init=tf.constant_initializer(b),
                    name='mean_std_network',
                    layer_normalization=self._layer_normalization)
                with tf.compat.v1.variable_scope('mean_network'):
                    mean_network = mean_std_network[..., :action_dim]
                with tf.compat.v1.variable_scope('log_std_network'):
                    log_std_network = mean_std_network[..., action_dim:]

            else:
                # separate MLPs for mean and std networks
                # mean network
                mean_conv = cnn(input_var=state_input,
                                filter_dims=self._filter_dims,
                                hidden_nonlinearity=self._hidden_nonlinearity,
                                hidden_w_init=self._hidden_w_init,
                                hidden_b_init=self._hidden_b_init,
                                num_filters=self._num_filters,
                                strides=self._strides,
                                padding=self._padding,
                                name='mean_cnn')

                mean_network = mlp(
                    mean_conv,
                    output_dim=action_dim,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    hidden_w_init=self._hidden_w_init,
                    hidden_b_init=self._hidden_b_init,
                    output_nonlinearity=self._output_nonlinearity,
                    output_w_init=self._output_w_init,
                    output_b_init=self._output_b_init,
                    name='mean_network',
                    layer_normalization=self._layer_normalization)

                # std network
                if self._adaptive_std:
                    log_std_conv = cnn(
                        input_var=state_input,
                        filter_dims=self._std_filter_dims,
                        hidden_nonlinearity=self._std_hidden_nonlinearity,
                        hidden_w_init=self._std_hidden_w_init,
                        hidden_b_init=self._std_hidden_b_init,
                        num_filters=self._std_num_filters,
                        strides=self._std_strides,
                        padding=self._std_padding,
                        name='log_std_cnn')

                    log_std_network = mlp(
                        log_std_conv,
                        output_dim=action_dim,
                        hidden_sizes=self._std_hidden_sizes,
                        hidden_nonlinearity=self._std_hidden_nonlinearity,
                        hidden_w_init=self._std_hidden_w_init,
                        hidden_b_init=self._std_hidden_b_init,
                        output_nonlinearity=self._std_output_nonlinearity,
                        output_w_init=self._std_output_w_init,
                        output_b_init=tf.constant_initializer(
                            self._init_std_param),
                        name='log_std_network',
                        layer_normalization=self._layer_normalization)
                else:
                    log_std_network = parameter(
                        input_var=state_input,
                        length=action_dim,
                        initializer=tf.constant_initializer(
                            self._init_std_param),
                        trainable=self._learn_std,
                        name='log_std_network')

        mean_var = mean_network
        std_param = log_std_network

        with tf.compat.v1.variable_scope('std_limits'):
            if self._min_std_param is not None:
                std_param = tf.maximum(std_param, self._min_std_param)
            if self._max_std_param is not None:
                std_param = tf.minimum(std_param, self._max_std_param)

        with tf.compat.v1.variable_scope('std_parameterization'):
            # build std_var with std parameterization
            if self._std_parameterization == 'exp':
                log_std_var = std_param
            else:  # we know it must be softplus here
                log_std_var = tf.math.log(tf.math.log(1. + tf.exp(std_param)))

        dist = DiagonalGaussian(self._output_dim)
        rnd = tf.random.normal(shape=mean_var.get_shape().as_list()[1:])
        action_var = rnd * tf.exp(log_std_var) + mean_var

        return action_var, mean_var, log_std_var, std_param, dist
