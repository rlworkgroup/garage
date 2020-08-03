"""Categorical LSTM Model.

A model represented by a Categorical distribution
which is parameterized by a Long short-term memory (LSTM).
"""
import tensorflow as tf
import tensorflow_probability as tfp

from garage.experiment import deterministic
from garage.tf.models.lstm_model import LSTMModel


class CategoricalLSTMModel(LSTMModel):
    """Categorical LSTM Model.

    A model represented by a Categorical distribution
    which is parameterized by a Long short-term memory (LSTM).

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
        super().__init__(
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            name=name,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            recurrent_nonlinearity=recurrent_nonlinearity,
            recurrent_w_init=recurrent_w_init,
            output_nonlinearity=tf.nn.softmax,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            hidden_state_init=hidden_state_init,
            hidden_state_init_trainable=hidden_state_init_trainable,
            cell_state_init=cell_state_init,
            cell_state_init_trainable=cell_state_init_trainable,
            forget_bias=forget_bias,
            layer_normalization=layer_normalization)
        self._output_normalization_fn = output_nonlinearity

    def network_output_spec(self):
        """Network output spec.

        Returns:
            list[str]: Name of the model outputs, in order.

        """
        return [
            'dist', 'step_output', 'step_hidden', 'step_cell', 'init_hidden',
            'init_cell'
        ]

    # pylint: disable=arguments-differ
    def _build(self,
               state_input,
               step_input,
               step_hidden,
               step_cell,
               name=None):
        """Build model.

        Args:
            state_input (tf.Tensor): Entire time-series observation input,
                with shape :math:`(N, T, S^*)`.
            step_input (tf.Tensor): Single timestep observation input,
                with shape :math:`(N, S^*)`.
            step_hidden (tf.Tensor): Hidden state for step, with shape
                :math:`(N, S^*)`.
            step_cell (tf.Tensor): Cell state for step, with shape
                :math:`(N, S^*)`.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Returns:
            tfp.distributions.OneHotCategorical: Policy distribution.
            tf.Tensor: Step output, with shape :math:`(N, S^*)`
            tf.Tensor: Step hidden state, with shape :math:`(N, S^*)`
            tf.Tensor: Step cell state, with shape :math:`(N, S^*)`
            tf.Tensor: Initial hidden state, used to reset the hidden state
                when policy resets. Shape: :math:`(S^*)`
            tf.Tensor: Initial cell state, used to reset the cell state
                when policy resets. Shape: :math:`(S^*)`

        """
        (outputs, step_output, step_hidden, step_cell, init_hidden,
         init_cell) = super()._build(state_input,
                                     step_input,
                                     step_hidden,
                                     step_cell,
                                     name=name)
        if self._output_normalization_fn:
            outputs = self._output_normalization_fn(outputs)
        dist = tfp.distributions.OneHotCategorical(probs=outputs)
        return (dist, step_output, step_hidden, step_cell, init_hidden,
                init_cell)
