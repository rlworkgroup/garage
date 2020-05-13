"""Gaussian GRU Model.

A model represented by a Gaussian distribution
which is parameterized by a Gated Recurrent Unit (GRU).
"""
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from garage.tf.models.gru import gru
from garage.tf.models.model import Model
from garage.tf.models.parameter import recurrent_parameter


class GaussianGRUModel(Model):
    """Gaussian GRU Model.

    A model represented by a Gaussian distribution
    which is parameterized by a Gated Recurrent Unit (GRU).

    Args:
        output_dim (int): Output dimension of the model.
        hidden_dim (int): Hidden dimension for GRU cell for mean.
        name (str): Model name, also the variable scope.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        recurrent_nonlinearity (callable): Activation function for recurrent
            layers. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        recurrent_w_init (callable): Initializer function for the weight
            of recurrent layer(s). The function should return a
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
        hidden_state_init (callable): Initializer function for the
            initial hidden state. The functino should return a tf.Tensor.
        hidden_state_init_trainable (bool): Bool for whether the initial
            hidden state is trainable.
        learn_std (bool): Is std trainable.
        init_std (float): Initial value for std.
        std_share_network (bool): Boolean for whether mean and std share
            the same network.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 output_dim,
                 hidden_dim=32,
                 name='GaussianGRUModel',
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.initializers.glorot_uniform(),
                 hidden_b_init=tf.zeros_initializer(),
                 recurrent_nonlinearity=tf.nn.sigmoid,
                 recurrent_w_init=tf.initializers.glorot_uniform(),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(),
                 output_b_init=tf.zeros_initializer(),
                 hidden_state_init=tf.zeros_initializer(),
                 hidden_state_init_trainable=False,
                 learn_std=True,
                 init_std=1.0,
                 std_share_network=False,
                 layer_normalization=False):
        super().__init__(name)
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._recurrent_nonlinearity = recurrent_nonlinearity
        self._recurrent_w_init = recurrent_w_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._hidden_state_init = hidden_state_init
        self._hidden_state_init_trainable = hidden_state_init_trainable
        self._layer_normalization = layer_normalization
        self._learn_std = learn_std
        self._std_share_network = std_share_network
        # pylint: disable=assignment-from-no-return
        self._init_std_param = np.log(init_std)
        self._initialize()

    def _initialize(self):
        action_dim = self._output_dim
        self._mean_std_gru_cell = tf.keras.layers.GRUCell(
            units=self._hidden_dim,
            activation=self._hidden_nonlinearity,
            kernel_initializer=self._hidden_w_init,
            bias_initializer=self._hidden_b_init,
            recurrent_activation=self._recurrent_nonlinearity,
            recurrent_initializer=self._recurrent_w_init,
            name='mean_std_gru_layer')
        self._mean_gru_cell = tf.keras.layers.GRUCell(
            units=self._hidden_dim,
            activation=self._hidden_nonlinearity,
            kernel_initializer=self._hidden_w_init,
            bias_initializer=self._hidden_b_init,
            recurrent_activation=self._recurrent_nonlinearity,
            recurrent_initializer=self._recurrent_w_init,
            name='mean_gru_layer')
        self._mean_std_output_nonlinearity_layer = tf.keras.layers.Dense(
            units=action_dim * 2,
            activation=self._output_nonlinearity,
            kernel_initializer=self._output_w_init,
            bias_initializer=self._output_b_init,
            name='mean_std_output_layer')
        self._mean_output_nonlinearity_layer = tf.keras.layers.Dense(
            units=action_dim,
            activation=self._output_nonlinearity,
            kernel_initializer=self._output_w_init,
            bias_initializer=self._output_b_init,
            name='mean_output_layer')

    def network_input_spec(self):
        """Network input spec.

        Returns:
            list[str]: Name of the model inputs, in order.

        """
        return ['full_input', 'step_input', 'step_hidden_input']

    def network_output_spec(self):
        """Network output spec.

        Returns:
            list[str]: Name of the model outputs, in order.

        """
        return [
            'dist', 'step_mean', 'step_log_std', 'step_hidden', 'init_hidden'
        ]

    # pylint: disable=arguments-differ
    def _build(self, state_input, step_input, step_hidden, name=None):
        """Build model.

        Args:
            state_input (tf.Tensor): Entire time-series observation input,
                with shape :math:`(N, T, S^*)`.
            step_input (tf.Tensor): Single timestep observation input,
                with shape :math:`(N, S^*)`.
            step_hidden (tf.Tensor): Hidden state for step, with shape
                :math:`(N, S^*)`.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Returns:
            tfp.distributions.MultivariateNormalDiag: Policy distribution.
            tf.Tensor: Step means, with shape :math:`(N, S^*)`.
            tf.Tensor: Step log std, with shape :math:`(N, S^*)`.
            tf.Tensor: Step hidden state, with shape :math:`(N, S^*)`.
            tf.Tensor: Initial hidden state, with shape :math:`(S^*)`.

        """
        del name
        action_dim = self._output_dim

        with tf.compat.v1.variable_scope('dist_params'):
            if self._std_share_network:
                # mean and std networks share an MLP
                (outputs, step_outputs, step_hidden, hidden_init_var) = gru(
                    name='mean_std_network',
                    gru_cell=self._mean_std_gru_cell,
                    all_input_var=state_input,
                    step_input_var=step_input,
                    step_hidden_var=step_hidden,
                    hidden_state_init=self._hidden_state_init,
                    hidden_state_init_trainable=self.
                    _hidden_state_init_trainable,
                    output_nonlinearity_layer=self.
                    _mean_std_output_nonlinearity_layer)
                with tf.compat.v1.variable_scope('mean_network'):
                    mean_var = outputs[..., :action_dim]
                    step_mean_var = step_outputs[..., :action_dim]
                with tf.compat.v1.variable_scope('log_std_network'):
                    log_std_var = outputs[..., action_dim:]
                    step_log_std_var = step_outputs[..., action_dim:]

            else:
                # separate MLPs for mean and std networks
                # mean network
                (mean_var, step_mean_var, step_hidden, hidden_init_var) = gru(
                    name='mean_network',
                    gru_cell=self._mean_gru_cell,
                    all_input_var=state_input,
                    step_input_var=step_input,
                    step_hidden_var=step_hidden,
                    hidden_state_init=self._hidden_state_init,
                    hidden_state_init_trainable=self.
                    _hidden_state_init_trainable,
                    output_nonlinearity_layer=self.
                    _mean_output_nonlinearity_layer)
                log_std_var, step_log_std_var = recurrent_parameter(
                    input_var=state_input,
                    step_input_var=step_input,
                    length=action_dim,
                    initializer=tf.constant_initializer(self._init_std_param),
                    trainable=self._learn_std,
                    name='log_std_param')

        dist = tfp.distributions.MultivariateNormalDiag(
            loc=mean_var, scale_diag=tf.exp(log_std_var))

        return (dist, step_mean_var, step_log_std_var, step_hidden,
                hidden_init_var)

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: the state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_mean_std_gru_cell']
        del new_dict['_mean_gru_cell']
        del new_dict['_mean_std_output_nonlinearity_layer']
        del new_dict['_mean_output_nonlinearity_layer']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
