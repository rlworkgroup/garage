"""Discrete MLP QFunction."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.core.mlp import mlp
from garage.tf.q_functions import QFunction


class DiscreteMLPQFunction(QFunction):
    """
    Discrete MLP Function.

    This class implements a Q-value network. It predicts Q-value based on the
    input state and action. It uses an MLP to fit the function Q(s, a).

    Args:
        env_spec: environment specification
        hidden_sizes: A list of numbers of hidden units
            for all hidden layers.
        hidden_nonlinearity: An activation shared by all fc layers.
        output_nonlinearity: An activation used by the output layer.
        layer_norm: A bool to indicate whether to perform
            layer normalization or not.
    """

    def __init__(self,
                 env_spec,
                 name="discrete_mlp_q_function",
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 layer_norm=False):
        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_norm = layer_norm

        with tf.name_scope(name):
            self.q_val, self.obs_ph = self.build_net(name)

    def _build_ph(self, scope):
        obs_dim = self._env_spec.observation_space.shape

        with tf.name_scope(scope):
            obs_ph = tf.placeholder(tf.float32, (None, ) + obs_dim, name="obs")

        return obs_ph

    @overrides
    def build_net(self, name):
        """
        Set up q network based on class attributes.

        Args:
            name: Network variable scope.
            input: Input tf.placeholder to the network.
        """
        obs_ph = self._build_ph(name)

        network = mlp(
            input_var=obs_ph,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            name=name,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=self._layer_norm)

        return network, obs_ph

    @overrides
    def get_qval_sym(self, input_phs):
        assert len(input_phs) == 1

        return mlp(
            input_var=input_phs,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            name=self.name,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=self._layer_norm,
            reuse=True)
