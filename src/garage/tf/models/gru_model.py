"""GRU Model.

A model composed only of a Gated Recurrent Unit (GRU).
"""
import tensorflow as tf

from garage.tf.models.base import Model
from garage.tf.models.gru import gru


class GRUModel(Model):
    """GRU Model.

    Args:
        output_dim (int): Dimension of the network output.
        hidden_dim (int): Hidden dimension for GRU cell.
        name (str): Policy name, also the variable scope.
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
        layer_normalization (bool): Bool for using layer normalization or not.
    """

    def __init__(self,
                 output_dim,
                 hidden_dim,
                 name=None,
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.glorot_uniform_initializer(),
                 hidden_b_init=tf.zeros_initializer(),
                 recurrent_nonlinearity=tf.nn.sigmoid,
                 recurrent_w_init=tf.glorot_uniform_initializer(),
                 output_nonlinearity=None,
                 output_w_init=tf.glorot_uniform_initializer(),
                 output_b_init=tf.zeros_initializer(),
                 hidden_state_init=tf.zeros_initializer(),
                 hidden_state_init_trainable=False,
                 layer_normalization=False):
        super().__init__(name)
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._recurrent_nonlinearity = recurrent_nonlinearity
        self._recurrent_w_init = recurrent_w_init
        self._hidden_state_init = hidden_state_init
        self._hidden_state_init_trainable = hidden_state_init_trainable
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._initialize()

    def _initialize(self):
        self._gru_cell = tf.keras.layers.GRUCell(
            units=self._hidden_dim,
            activation=self._hidden_nonlinearity,
            kernel_initializer=self._hidden_w_init,
            bias_initializer=self._hidden_b_init,
            recurrent_activation=self._recurrent_nonlinearity,
            recurrent_initializer=self._recurrent_w_init,
            name='gru_layer')
        self._output_nonlinearity_layer = tf.keras.layers.Dense(
            units=self._output_dim,
            activation=self._output_nonlinearity,
            kernel_initializer=self._output_w_init,
            bias_initializer=self._output_b_init,
            name='output_layer')

    def network_input_spec(self):
        """Network input spec."""
        return ['full_input', 'step_input', 'step_hidden_input']

    def network_output_spec(self):
        """Network output spec."""
        return ['all_output', 'step_output', 'step_hidden', 'init_hidden']

    def _build(self, all_input_var, step_input_var, step_hidden_var,
               name=None):
        return gru(
            name='gru',
            gru_cell=self._gru_cell,
            all_input_var=all_input_var,
            step_input_var=step_input_var,
            step_hidden_var=step_hidden_var,
            hidden_state_init=self._hidden_state_init,
            hidden_state_init_trainable=self._hidden_state_init_trainable,
            output_nonlinearity_layer=self._output_nonlinearity_layer)

    def __getstate__(self):
        """Object.__getstate__."""
        new_dict = super().__getstate__()
        del new_dict['_gru_cell']
        del new_dict['_output_nonlinearity_layer']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__."""
        super().__setstate__(state)
        self._initialize()
