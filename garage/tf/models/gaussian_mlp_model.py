"""GaussianMLPModel."""
import numpy as np
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.core.mlp import mlp
from garage.tf.core.parameter import parameter
from garage.tf.models.base import Model


class GaussianMLPModel(Model):
    """
    GaussianMLPModel.

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
                 output_dim,
                 name=None,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.tanh,
                 output_nonlinearity=None,
                 learn_std=True,
                 adaptive_std=False,
                 std_share_network=False,
                 init_std=1.0,
                 min_std=1e-6,
                 max_std=None,
                 std_hidden_sizes=(32, 32),
                 std_hidden_nonlinearity=tf.nn.tanh,
                 std_output_nonlinearity=None,
                 std_parameterization='exp',
                 layer_normalization=False):
        # Network parameters
        super().__init__(name)
        self._hidden_sizes = hidden_sizes
        self._output_dim = output_dim
        self._learn_std = learn_std
        self._adaptive_std = adaptive_std
        self._std_share_network = std_share_network
        self._std_hidden_sizes = std_hidden_sizes
        self._min_std = min_std
        self._max_std = max_std
        self._std_hidden_nonlinearity = std_hidden_nonlinearity
        self._std_output_nonlinearity = std_output_nonlinearity
        self._std_parameterization = std_parameterization
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_normalization = layer_normalization

        # Tranform std arguments to parameterized space
        self._init_std_param = None
        self._min_std_param = None
        self._max_std_param = None
        if self._std_parameterization == 'exp':
            self._init_std_param = np.log(init_std)
            if min_std:
                self._min_std_param = np.log(min_std)
            if max_std:
                self._max_std_param = np.log(max_std)
        elif self._std_parameterization == 'softplus':
            self._init_std_param = np.log(np.exp(init_std) - 1)
            if min_std:
                self._min_std_param = np.log(np.exp(min_std) - 1)
            if max_std:
                self._max_std_param = np.log(np.exp(max_std) - 1)
        else:
            raise NotImplementedError

    def network_output_spec(self):
        """Network output spec."""
        return ['sample', 'mean', 'std', 'std_param']

    def _build(self, state_input):
        action_dim = self._output_dim

        with tf.variable_scope('dist_params'):
            if self._std_share_network:
                # mean and std networks share an MLP
                b = np.concatenate([
                    np.zeros(action_dim),
                    np.full(action_dim, self._init_std_param)
                ], axis=0)  # yapf: disable
                b = tf.constant_initializer(b)
                mean_std_network = mlp(
                    state_input,
                    output_dim=action_dim * 2,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._output_nonlinearity,
                    output_b_init=b,
                    name='mean_std_network',
                    layer_normalization=self._layer_normalization)
                with tf.variable_scope('mean_network'):
                    mean_network = mean_std_network[..., :action_dim]
                with tf.variable_scope('std_network'):
                    std_network = mean_std_network[..., action_dim:]

            else:
                # separate MLPs for mean and std networks
                # mean network
                mean_network = mlp(
                    state_input,
                    output_dim=action_dim,
                    hidden_sizes=self._hidden_sizes,
                    hidden_nonlinearity=self._hidden_nonlinearity,
                    output_nonlinearity=self._output_nonlinearity,
                    name='mean_network',
                    layer_normalization=self._layer_normalization)

                # std network
                if self._adaptive_std:
                    b = tf.constant_initializer(self._init_std_param)
                    std_network = mlp(
                        state_input,
                        output_dim=action_dim,
                        hidden_sizes=self._std_hidden_sizes,
                        hidden_nonlinearity=self._std_hidden_nonlinearity,
                        output_nonlinearity=self._std_output_nonlinearity,
                        output_b_init=b,
                        name='std_network',
                        layer_normalization=self._layer_normalization)
                else:
                    p = tf.constant_initializer(self._init_std_param)
                    std_network = parameter(
                        state_input,
                        length=action_dim,
                        initializer=p,
                        trainable=self._learn_std,
                        name='std_network')

        mean_var = mean_network
        std_param_var = std_network

        with tf.variable_scope('std_parameterization'):
            # build std_var with std parameterization
            if self._std_parameterization == 'exp':
                std_param_var = std_param_var
            elif self._std_parameterization == 'softplus':
                std_param_var = tf.log(1. + tf.exp(std_param_var))
            else:
                raise NotImplementedError

        with tf.variable_scope('std_limits'):
            if self._min_std_param:
                std_var = tf.maximum(std_param_var, self._min_std_param)
            if self._max_std_param:
                std_var = tf.minimum(std_param_var, self._max_std_param)

        rnd = tf.random.normal(
            shape=mean_var.get_shape().as_list()[1:],
            seed=deterministic.get_seed())
        action_var = rnd * tf.exp(std_var) + mean_var

        return action_var, mean_var, std_var, std_param_var
