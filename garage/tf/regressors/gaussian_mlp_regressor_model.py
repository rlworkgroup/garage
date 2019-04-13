"""GaussianMLPRegressorModel."""
import numpy as np
import tensorflow as tf

from garage.tf.models import GaussianMLPModel


class GaussianMLPRegressorModel(GaussianMLPModel):
    """
    GaussianMLPRegressor based on garage.tf.models.Model class.

    This class can be used to perform regression by fitting a Gaussian
    distribution to the outputs.

    Args:
    :param output_dim: Output dimension of the model.
    :param name: Name of the model.
    :param hidden_sizes: List of sizes for the fully-connected hidden
        layers.
    :param learn_std: Is std trainable.
    :param init_std: Initial value for std.
    :param adaptive_std: Is std a neural network. If False, it will be a
        parameter.
    :param std_share_network: Boolean for whether mean and std share the same
        network.
    :param std_hidden_sizes: List of sizes for the fully-connected layers
        for std.
    :param min_std: Whether to make sure that the std is at least some
        threshold value, to avoid numerical issues.
    :param max_std: Whether to make sure that the std is at most some
        threshold value, to avoid numerical issues.
    :param std_hidden_nonlinearity: Nonlinearity for each hidden layer in
        the std network.
    :param std_output_nonlinearity: Nonlinearity for output layer in
        the std network.
    :param std_parametrization: How the std should be parametrized. There
        are a few options:
        - exp: the logarithm of the std will be stored, and applied a
            exponential transformation
        - softplus: the std will be computed as log(1+exp(x))
    :param hidden_nonlinearity: Nonlinearity used for each hidden layer.
    :param output_nonlinearity: Nonlinearity for the output layer.
    """

    def __init__(self,
                 input_shape,
                 output_dim,
                 name='GaussianMLPRegressorModel',
                 **kwargs):
        # Network parameters
        super().__init__(output_dim=output_dim, name=name, **kwargs)
        self._input_shape = input_shape

    def network_output_spec(self):
        """Network output spec."""
        return [
            'sample', 'means', 'log_stds', 'std_param', 'normalized_means',
            'normalized_log_stds', 'x_mean', 'x_std', 'y_mean', 'y_std', 'dist'
        ]

    def _build(self, state_input, name=None):
        with tf.variable_scope('normalized_vars'):
            x_mean_var = tf.get_variable(
                name='x_mean',
                shape=(1, ) + self._input_shape,
                dtype=np.float32,
                initializer=tf.zeros_initializer(),
                trainable=False)
            x_std_var = tf.get_variable(
                name='x_std_var',
                shape=(1, ) + self._input_shape,
                dtype=np.float32,
                initializer=tf.ones_initializer(),
                trainable=False)
            y_mean_var = tf.get_variable(
                name='y_mean_var',
                shape=(1, self._output_dim),
                dtype=np.float32,
                initializer=tf.zeros_initializer(),
                trainable=False)
            y_std_var = tf.get_variable(
                name='y_std_var',
                shape=(1, self._output_dim),
                dtype=np.float32,
                initializer=tf.ones_initializer(),
                trainable=False)

        normalized_xs_var = (state_input - x_mean_var) / x_std_var

        sample, normalized_mean, normalized_log_std, std_param, dist = super(
        )._build(normalized_xs_var)

        with tf.name_scope('mean_network'):
            means_var = normalized_mean * y_std_var + y_mean_var

        with tf.name_scope('std_network'):
            log_stds_var = normalized_log_std + tf.log(y_std_var)

        return (sample, means_var, log_stds_var, std_param, normalized_mean,
                normalized_log_std, x_mean_var, x_std_var, y_mean_var,
                y_std_var, dist)
