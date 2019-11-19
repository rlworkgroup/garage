"""LSTM in TensorFlow."""
import tensorflow as tf


def lstm(name,
         lstm_cell,
         all_input_var,
         step_input_var,
         step_hidden_var,
         step_cell_var,
         output_nonlinearity_layer,
         hidden_state_init=tf.zeros_initializer(),
         hidden_state_init_trainable=False,
         cell_state_init=tf.zeros_initializer(),
         cell_state_init_trainable=False):
    """Long Short-Term Memory (LSTM).

    Args:
        name (str): Name of the variable scope.
        lstm_cell (tf.keras.layers.Layer): LSTM cell used to generate
            outputs.
        all_input_var (tf.Tensor): Place holder for entire time-seried inputs.
        step_input_var (tf.Tensor): Place holder for step inputs.
        step_hidden_var (tf.Tensor): Place holder for step hidden state.
        step_cell_var (tf.Tensor): Place holder for cell state.
        output_nonlinearity_layer (callable): Activation function for output
            dense layer. It should return a tf.Tensor. Set it to None to
            maintain a linear activation.
        hidden_state_init (callable): Initializer function for the
            initial hidden state. The functino should return a tf.Tensor.
        hidden_state_init_trainable (bool): Bool for whether the initial
            hidden state is trainable.
        cell_state_init (callable): Initializer function for the
            initial cell state. The functino should return a tf.Tensor.
        cell_state_init_trainable (bool): Bool for whether the initial
            cell state is trainable.

    Return:
        outputs (tf.Tensor): Entire time-seried outputs.
        output (tf.Tensor): Step output.
        hidden (tf.Tensor): Step hidden state.
        cell (tf.Tensor): Step cell state.
        hidden_init_var (tf.Tensor): Initial hidden state.
        cell_init_var (tf.Tensor): Initial cell state.

    """
    with tf.compat.v1.variable_scope(name):
        hidden_dim = lstm_cell.units
        output, [hidden,
                 cell] = lstm_cell(step_input_var,
                                   states=(step_hidden_var, step_cell_var))
        output = output_nonlinearity_layer(output)

        hidden_init_var = tf.compat.v1.get_variable(
            name='initial_hidden',
            shape=(hidden_dim, ),
            initializer=hidden_state_init,
            trainable=hidden_state_init_trainable,
            dtype=tf.float32)
        cell_init_var = tf.compat.v1.get_variable(
            name='initial_cell',
            shape=(hidden_dim, ),
            initializer=cell_state_init,
            trainable=cell_state_init_trainable,
            dtype=tf.float32)

        hidden_init_var_b = tf.broadcast_to(
            hidden_init_var, shape=[tf.shape(all_input_var)[0], hidden_dim])
        cell_init_var_b = tf.broadcast_to(
            cell_init_var, shape=[tf.shape(all_input_var)[0], hidden_dim])

        def step(hcprev, x):
            hprev = hcprev[:, :hidden_dim]
            cprev = hcprev[:, hidden_dim:]
            h, c = lstm_cell(x, states=(hprev, cprev))[1]
            return tf.concat(axis=1, values=[h, c])

        shuffled_input = tf.transpose(all_input_var, (1, 0, 2))
        hcs = tf.scan(
            step,
            elems=shuffled_input,
            initializer=tf.concat(axis=1,
                                  values=[hidden_init_var_b, cell_init_var_b]),
        )
        hcs = tf.transpose(hcs, (1, 0, 2))
        hs = hcs[:, :, :hidden_dim]
        outputs = output_nonlinearity_layer(hs)

    return outputs, output, hidden, cell, hidden_init_var, cell_init_var
