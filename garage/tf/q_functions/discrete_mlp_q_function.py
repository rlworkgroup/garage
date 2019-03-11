"""Discrete MLP QFunction."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.models.mlp_model import MLPModel


class DiscreteMLPQFunction:
    """
    Discrete MLP Q Function.

    This class implements a Q-value network. It predicts Q-value based on the
    input state and action. It uses an MLP to fit the function Q(s, a).

    Args:
        env_spec: Environment specification.
        name: Name of the q-function, also serves as the variable scope.
        hidden_sizes: Output dimension of dense layer(s).
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).
        layer_normalization: Bool for using layer normalization or not.
    """

    def __init__(self,
                 env_spec,
                 name="discrete_mlp_q_function",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.contrib.layers.xavier_initializer,
                 hidden_b_init=tf.zeros_initializer,
                 output_nonlinearity=None,
                 output_w_init=tf.contrib.layers.xavier_initializer,
                 output_b_init=tf.zeros_initializer,
                 layer_normalization=False):
        obs_dim = env_spec.observation_space.shape
        action_dim = env_spec.action_space.flat_dim

        self._variable_scope = tf.VariableScope(reuse=False, name=name)
        self.model = MLPModel(
            output_dim=action_dim,
            name=name,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=output_nonlinearity,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

        obs_ph = tf.placeholder(tf.float32, (None, ) + obs_dim, name="obs")

        with tf.variable_scope(self._variable_scope):
            self.model.build(obs_ph)

    @overrides
    def get_qval_sym(self, state_input, name):
        """
        Symbolic graph for q-network.

        Args:
            state_input: The state input tf.Tensor to the network.
            name: Network variable scope.
        """
        with tf.variable_scope(self._variable_scope):
            return self.model.build(state_input, name=name)
