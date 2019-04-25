"""MLP in TensorFlow."""

import tensorflow as tf


def mlp(input_var,
        output_dim,
        hidden_sizes,
        name,
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.glorot_uniform_initializer(),
        hidden_b_init=tf.zeros_initializer(),
        output_nonlinearity=None,
        output_w_init=tf.glorot_uniform_initializer(),
        output_b_init=tf.zeros_initializer(),
        layer_normalization=False):
    """
    Multi-layer perceptron (MLP).

    It maps real-valued inputs to real-valued outputs.

    Args:
        input_var (tf.Tensor): Input tf.Tensor to the MLP.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        name (str): Network name, also the variable scope.
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

    Return:
        The output tf.Tensor of the MLP
    """
    with tf.variable_scope(name):
        l_hid = input_var
        for idx, hidden_size in enumerate(hidden_sizes):
            l_hid = tf.layers.dense(
                inputs=l_hid,
                units=hidden_size,
                activation=hidden_nonlinearity,
                kernel_initializer=hidden_w_init,
                bias_initializer=hidden_b_init,
                name='hidden_{}'.format(idx))
            if layer_normalization:
                l_hid = tf.contrib.layers.layer_norm(l_hid)

        l_out = tf.layers.dense(
            inputs=l_hid,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name='output')
    return l_out
