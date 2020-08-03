"""LSTM Model.

A model composed only of a long-short term memory (LSTM).
"""
import tensorflow as tf

from garage.experiment import deterministic
from garage.tf.models.lstm import lstm
from garage.tf.models.model import Model


class LSTMModel(Model):
    """LSTM Model.

    Args:
        output_dim (int): Dimension of the network output.
        hidden_dim (int): Hidden dimension for LSTM cell.
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
        cell_state_init (callable): Initializer function for the
            initial cell state. The functino should return a tf.Tensor.
        cell_state_init_trainable (bool): Bool for whether the initial
            cell state is trainable.
        forget_bias (bool): If True, add 1 to the bias of the forget gate at
            initialization. It's used to reduce the scale of forgetting at the
            beginning of the training.
        layer_normalization (bool): Bool for using layer normalization or not.

    """

    def __init__(self,
                 output_dim,
                 hidden_dim,
                 name=None,
                 hidden_nonlinearity=tf.nn.tanh,
                 hidden_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 hidden_b_init=tf.zeros_initializer(),
                 recurrent_nonlinearity=tf.nn.sigmoid,
                 recurrent_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_nonlinearity=None,
                 output_w_init=tf.initializers.glorot_uniform(
                     seed=deterministic.get_tf_seed_stream()),
                 output_b_init=tf.zeros_initializer(),
                 hidden_state_init=tf.zeros_initializer(),
                 hidden_state_init_trainable=False,
                 cell_state_init=tf.zeros_initializer(),
                 cell_state_init_trainable=False,
                 forget_bias=True,
                 layer_normalization=False):
        super().__init__(name)
        self._output_dim = output_dim
        self._hidden_dim = hidden_dim
        self._forget_bias = forget_bias
        self._hidden_nonlinearity = hidden_nonlinearity
        self._hidden_w_init = hidden_w_init
        self._hidden_b_init = hidden_b_init
        self._recurrent_nonlinearity = recurrent_nonlinearity
        self._recurrent_w_init = recurrent_w_init
        self._hidden_state_init = hidden_state_init
        self._hidden_state_init_trainable = hidden_state_init_trainable
        self._cell_state_init = cell_state_init
        self._cell_state_init_trainable = cell_state_init_trainable
        self._output_nonlinearity = output_nonlinearity
        self._output_w_init = output_w_init
        self._output_b_init = output_b_init
        self._layer_normalization = layer_normalization
        self._initialize()

    def _initialize(self):
        """Initialize model."""
        self._lstm_cell = tf.keras.layers.LSTMCell(
            units=self._hidden_dim,
            activation=self._hidden_nonlinearity,
            kernel_initializer=self._hidden_w_init,
            bias_initializer=self._hidden_b_init,
            recurrent_activation=self._recurrent_nonlinearity,
            recurrent_initializer=self._recurrent_w_init,
            unit_forget_bias=self._forget_bias,
            name='lstm_layer')
        self._output_nonlinearity_layer = tf.keras.layers.Dense(
            units=self._output_dim,
            activation=self._output_nonlinearity,
            kernel_initializer=self._output_w_init,
            bias_initializer=self._output_b_init,
            name='output_layer')

    def network_input_spec(self):
        """Network input spec.

        Return:
            list[str]: List of key(str) for the network outputs.

        """
        return [
            'full_input', 'step_input', 'step_hidden_input', 'step_cell_input'
        ]

    def network_output_spec(self):
        """Network output spec.

        Return:
            list[str]: List of key(str) for the network outputs.

        """
        return [
            'all_output', 'step_output', 'step_hidden', 'step_cell',
            'init_hidden', 'init_cell'
        ]

    # pylint: disable=arguments-differ
    def _build(self,
               all_input_var,
               step_input_var,
               step_hidden_var,
               step_cell_var,
               name=None):
        """Build model given input placeholder(s).

        Args:
            all_input_var (tf.Tensor): Place holder for entire time-series
                inputs.
            step_input_var (tf.Tensor): Place holder for step inputs.
            step_hidden_var (tf.Tensor): Place holder for step hidden state.
            step_cell_var (tf.Tensor): Place holder for step cell state.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Return:
            tf.Tensor: Entire time-series outputs.
            tf.Tensor: Step output.
            tf.Tensor: Step hidden state.
            tf.Tensor: Step cell state.
            tf.Tensor: Initial hidden state.
            tf.Tensor: Initial cell state.

        """
        del name
        return lstm(
            name='lstm',
            lstm_cell=self._lstm_cell,
            all_input_var=all_input_var,
            step_input_var=step_input_var,
            step_hidden_var=step_hidden_var,
            step_cell_var=step_cell_var,
            hidden_state_init=self._hidden_state_init,
            hidden_state_init_trainable=self._hidden_state_init_trainable,
            cell_state_init=self._cell_state_init,
            cell_state_init_trainable=self._cell_state_init_trainable,
            output_nonlinearity_layer=self._output_nonlinearity_layer)

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = super().__getstate__()
        del new_dict['_lstm_cell']
        del new_dict['_output_nonlinearity_layer']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
