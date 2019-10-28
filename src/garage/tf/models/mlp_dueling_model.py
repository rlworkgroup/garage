"""MLP Dueling Model."""
import tensorflow as tf

from garage.tf.models.base import Model
from garage.tf.models.mlp import mlp


class MLPDuelingModel(Model):
    """MLP Model with dueling network structure.

    Args:
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
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
        output_nonlinearity (callable): Activation function for output dense
            layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            tf.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            tf.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 output_dim,
                 name=None,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=tf.nn.relu,
                 hidden_w_init=tf.contrib.layers.xavier_initializer,
                 hidden_b_init=tf.zeros_initializer,
                 output_nonlinearity=None,
                 output_w_init=tf.contrib.layers.xavier_initializer,
                 output_b_init=tf.zeros_initializer,
                 layer_normalization=False):
        super().__init__(name)
        self._output_dim = output_dim
        self._hidden_sizes = hidden_sizes
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization

    def _build(self, state_input, name=None):
        action_out = mlp(input_var=state_input,
                         output_dim=self._output_dim,
                         hidden_sizes=self._hidden_sizes,
                         name='action_value',
                         hidden_nonlinearity=self._hidden_nonlinearity,
                         hidden_w_init=self._hidden_w_init,
                         hidden_b_init=self._hidden_b_init,
                         output_nonlinearity=self._output_nonlinearity,
                         output_w_init=self._output_w_init,
                         output_b_init=self._output_b_init,
                         layer_normalization=self._layer_normalization)
        state_out = mlp(input_var=state_input,
                        output_dim=1,
                        hidden_sizes=self._hidden_sizes,
                        name='state_value',
                        hidden_nonlinearity=self._hidden_nonlinearity,
                        hidden_w_init=self._hidden_w_init,
                        hidden_b_init=self._hidden_b_init,
                        output_nonlinearity=self._output_nonlinearity,
                        output_w_init=self._output_w_init,
                        output_b_init=self._output_b_init,
                        layer_normalization=self._layer_normalization)

        action_out_mean = tf.reduce_mean(action_out, 1)
        # calculate the advantage of performing certain action
        # over other action in a particular state
        action_out_advantage = action_out - tf.expand_dims(action_out_mean, 1)
        q_func_out = state_out + action_out_advantage

        return q_func_out
