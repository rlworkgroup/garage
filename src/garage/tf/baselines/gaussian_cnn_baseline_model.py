"""GaussianCNNRegressorModel."""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from garage.experiment import deterministic
from garage.tf.models import GaussianCNNModel


class GaussianCNNBaselineModel(GaussianCNNModel):
    """GaussianCNNBaseline based on garage.tf.models.Model class.

    This class can be used to perform regression by fitting a Gaussian
    distribution to the outputs.

    Args:
        input_shape(tuple[int]): Input shape of the model (without the batch
            dimension).
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
        min_std (float): If not None, the std is at least the value of min_std,
            to avoid numerical issues.
        max_std (float): If not None, the std is at most the value of max_std,
            to avoid numerical issues.
        std_hidden_nonlinearity (callable): Nonlinearity for each hidden layer
            in the std network.
        std_hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s) in the std network.
        std_hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s) in the std network.
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
                 input_shape,
                 output_dim,
                 filters,
                 strides,
                 padding,
                 hidden_sizes,
                 name='GaussianCNNRegressorModel',
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 learn_std=True,
                 adaptive_std=False,
                 std_share_network=False,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_filters=(),
                 std_strides=(),
                 std_padding='SAME',
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=tf.nn.tanh,
                 std_hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 std_hidden_b_init=tf.zeros_initializer(),
                 std_output_nonlinearity=None,
                 std_output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 std_parameterization='exp',
                 layer_normalization=False):
        super().__init__(output_dim=output_dim,
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
                         min_std=min_std,
                         max_std=max_std,
                         std_filters=std_filters,
                         std_strides=std_strides,
                         std_padding=std_padding,
                         std_hidden_sizes=std_hidden_sizes,
                         std_hidden_nonlinearity=std_hidden_nonlinearity,
                         std_hidden_w_init=std_hidden_w_init,
                         std_hidden_b_init=std_hidden_b_init,
                         std_output_nonlinearity=std_output_nonlinearity,
                         std_output_w_init=std_output_w_init,
                         std_parameterization=std_parameterization,
                         layer_normalization=layer_normalization,
                         name=name)
        self._input_shape = input_shape

    def network_output_spec(self):
        """Network output spec.

        Return:
            list[str]: List of key(str) for the network outputs.

        """
        return [
            'sample', 'std_param', 'normalized_dist', 'normalized_mean',
            'normalized_log_std', 'dist', 'mean', 'log_std', 'x_mean', 'x_std',
            'y_mean', 'y_std'
        ]

    def _build(self, state_input, name=None):
        """Build model given input placeholder(s).

        Args:
            state_input (tf.Tensor): Place holder for state input.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Return:
            tf.Tensor: Sampled action.
            tf.Tensor: Parameterized log_std.
            tfp.distributions.MultivariateNormalDiag: Normlizaed distribution.
            tf.Tensor: Normalized mean.
            tf.Tensor: Normalized log_std.
            tfp.distributions.MultivariateNormalDiag: Vanilla distribution.
            tf.Tensor: Vanilla mean.
            tf.Tensor: Vanilla log_std.
            tf.Tensor: Mean for data.
            tf.Tensor: log_std for data.
            tf.Tensor: Mean for label.
            tf.Tensor: log_std for label.

        """
        with tf.compat.v1.variable_scope('normalized_vars'):
            x_mean_var = tf.compat.v1.get_variable(
                name='x_mean',
                shape=(1, ) + self._input_shape,
                dtype=np.float32,
                initializer=tf.zeros_initializer(),
                trainable=False)
            x_std_var = tf.compat.v1.get_variable(
                name='x_std_var',
                shape=(1, ) + self._input_shape,
                dtype=np.float32,
                initializer=tf.ones_initializer(),
                trainable=False)
            y_mean_var = tf.compat.v1.get_variable(
                name='y_mean_var',
                shape=(1, self._output_dim),
                dtype=np.float32,
                initializer=tf.zeros_initializer(),
                trainable=False)
            y_std_var = tf.compat.v1.get_variable(
                name='y_std_var',
                shape=(1, self._output_dim),
                dtype=np.float32,
                initializer=tf.ones_initializer(),
                trainable=False)

        normalized_xs_var = (state_input - x_mean_var) / x_std_var

        (sample, normalized_dist_mean, normalized_dist_log_std, std_param,
         _) = super()._build(normalized_xs_var)

        with tf.name_scope('mean_network'):
            means_var = normalized_dist_mean * y_std_var + y_mean_var

        with tf.name_scope('std_network'):
            log_stds_var = normalized_dist_log_std + tf.math.log(y_std_var)

        normalized_dist = tfp.distributions.MultivariateNormalDiag(
            loc=normalized_dist_mean,
            scale_diag=tf.exp(normalized_dist_log_std))

        vanilla_dist = tfp.distributions.MultivariateNormalDiag(
            loc=means_var, scale_diag=tf.exp(log_stds_var))

        return (sample, std_param, normalized_dist, normalized_dist_mean,
                normalized_dist_log_std, vanilla_dist, means_var, log_stds_var,
                x_mean_var, x_std_var, y_mean_var, y_std_var)
