"""GRU in TensorFlow."""
import tensorflow as tf


def gru(name,
        gru_cell,
        all_input_var,
        step_input_var,
        step_hidden_var,
        output_nonlinearity_layer,
        hidden_state_init=tf.zeros_initializer(),
        hidden_state_init_trainable=False):
    r"""Gated Recurrent Unit (GRU).

    Args:
        name (str): Name of the variable scope.
        gru_cell (tf.keras.layers.Layer): GRU cell used to generate
            outputs.
        all_input_var (tf.Tensor): Place holder for entire time-series inputs,
            with shape :math:`(N, T, S^*)`.
        step_input_var (tf.Tensor): Place holder for step inputs, with shape
            :math:`(N, S^*)`.
        step_hidden_var (tf.Tensor): Place holder for step hidden state, with
            shape :math:`(N, H)`.
        output_nonlinearity_layer (callable): Activation function for output
            dense layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        hidden_state_init (callable): Initializer function for the
            initial hidden state. The functino should return a tf.Tensor.
        hidden_state_init_trainable (bool): Bool for whether the initial
            hidden state is trainable.

    Return:
        tf.Tensor: Entire time-series outputs, with shape :math:`(N, T, S^*)`.
        tf.Tensor: Step output, with shape :math:`(N, S^*)`.
        tf.Tensor: Step hidden state, with shape :math:`(N, H)`
        tf.Tensor: Initial hidden state, with shape :math:`(H, )`

    """
    with tf.compat.v1.variable_scope(name):
        hidden_dim = gru_cell.units
        output, [hidden] = gru_cell(step_input_var, states=[step_hidden_var])
        output = output_nonlinearity_layer(output)

        hidden_init_var = tf.compat.v1.get_variable(
            name='initial_hidden',
            shape=(hidden_dim, ),
            initializer=hidden_state_init,
            trainable=hidden_state_init_trainable,
            dtype=tf.float32)

        hidden_init_var_b = tf.broadcast_to(
            hidden_init_var, shape=[tf.shape(all_input_var)[0], hidden_dim])

        rnn = tf.keras.layers.RNN(gru_cell, return_sequences=True)

        hs = rnn(all_input_var, initial_state=hidden_init_var_b)
        outputs = output_nonlinearity_layer(hs)

    return outputs, output, hidden, hidden_init_var
