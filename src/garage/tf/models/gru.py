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
    """Gated Recurrent Unit (GRU).

    Args:
        name (str): Name of the variable scope.
        gru_cell (tf.keras.layers.Layer): GRU cell used to generate
            outputs.
        all_input_var (tf.Tensor): Place holder for entire time-series inputs.
        step_input_var (tf.Tensor): Place holder for step inputs.
        step_hidden_var (tf.Tensor): Place holder for step hidden state.
        output_nonlinearity_layer (callable): Activation function for output
            dense layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        hidden_state_init (callable): Initializer function for the
            initial hidden state. The functino should return a tf.Tensor.
        hidden_state_init_trainable (bool): Bool for whether the initial
            hidden state is trainable.

    Return:
        outputs (tf.Tensor): Entire time-series outputs.
        output (tf.Tensor): Step output.
        hidden (tf.Tensor): Step hidden state.
        hidden_init_var (tf.Tensor): Initial hidden state.

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

        def step(hprev, x):
            _, [h] = gru_cell(x, states=[hprev])
            return h

        shuffled_input = tf.transpose(all_input_var, (1, 0, 2))
        hs = tf.scan(step, elems=shuffled_input, initializer=hidden_init_var_b)
        hs = tf.transpose(hs, (1, 0, 2))
        outputs = output_nonlinearity_layer(hs)

    return outputs, output, hidden, hidden_init_var
