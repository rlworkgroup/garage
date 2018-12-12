"""Discrete MLP QFunction."""
import tensorflow as tf

from garage.misc.overrides import overrides
from garage.tf.core.mlp import mlp
from garage.tf.q_functions import QFunction


class DiscreteMLPQFunction(QFunction):
    """
    Discrete MLP Function class.

    This class implements a Q-value network. It predicts Q-value based on the
    input state and action. It uses an MLP to fit the function Q(s, a).
    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 output_nonlinearity=None,
                 layer_norm=False):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec: environment specification
            hidden_sizes: A list of numbers of hidden units
                for all hidden layers.
            hidden_nonlinearity: An activation shared by all fc layers.
            output_nonlinearity: An activation used by the output layer.
            layer_norm: A bool to indicate whether to perform
                layer normalization or not.
        """
        super(DiscreteMLPQFunction, self).__init__()

        self._action_dim = env_spec.action_space.flat_dim
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._output_nonlinearity = output_nonlinearity
        self._layer_norm = layer_norm

    @overrides
    def build_net(self, name, input):
        """
        Set up q network based on class attributes.

        Args:
            name: Network variable scope.
            input: Input tf.placeholder to the network.
        """
        return mlp(
            input_var=input,
            output_dim=self._action_dim,
            hidden_sizes=self._hidden_sizes,
            name=name,
            hidden_nonlinearity=self._hidden_nonlinearity,
            output_nonlinearity=self._output_nonlinearity,
            layer_normalization=self._layer_norm)
