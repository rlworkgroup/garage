import tensorflow as tf


def mlp(input_var,
        output_dim,
        hidden_sizes,
        hidden_nonlinearity=tf.nn.relu,
        output_nonlinearity=None,
        name="mlp",
        hidden_w_init=tf.contrib.layers.xavier_initializer(),
        hidden_b_init=tf.zeros_initializer(),
        output_w_init=tf.contrib.layers.xavier_initializer(),
        output_b_init=tf.zeros_initializer(),
        batch_normalization=False,
        weight_normalization=False,
        is_training=True,
        debug=False):

    _layers = []

    with tf.variable_scope(name):
        l_hid = input_var
        _layers = [l_hid]
        if batch_normalization:
            l_hid = tf.layers.batch_normalization(l_hid, training=is_training)
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = tf.layers.dense(
                inputs=l_hid,
                units=hidden_size,
                activation=hidden_nonlinearity,
                kernel_initializer=hidden_w_init,
                bias_initializer=hidden_b_init,
                name="hidden_%d" % idx,
            )

            _layers.append(l_hid)
        l_out = tf.layers.dense(
            inputs=l_hid,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="output")
        if batch_normalization:
            l_out = tf.layers.batch_normalization(l_out, training=is_training)
        _layers.append(l_out)
        _l_out = l_out

    if debug:
        return _l_out, _layers
    else:
        return _l_out
