"""MLP in TensorFlow."""

import tensorflow as tf


def mlp(input_var,
        output_dim,
        hidden_sizes,
        name,
        input_var2=None,
        concat_layer=-2,
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.glorot_uniform_initializer(),
        hidden_b_init=tf.zeros_initializer(),
        output_nonlinearity=None,
        output_w_init=tf.glorot_uniform_initializer(),
        output_b_init=tf.zeros_initializer(),
        layer_normalization=False):
    """Multi-layer perceptron (MLP).

    It maps real-valued inputs to real-valued outputs.

    Args:
        input_var (tf.Tensor): Input tf.Tensor to the MLP.
        output_dim (int): Dimension of the network output.
        hidden_sizes (list[int]): Output dimension of dense layer(s).
            For example, (32, 32) means this MLP consists of two
            hidden layers, each with 32 hidden units.
        name (str): Network name, also the variable scope.
        input_var2 (tf.Tensor): Second input tf.Tensor to the MLP if input
            needs to be concatenated with a layer in the model.
        concat_layer (int): The index of layers at which to concatenate
            input_var2 with the network. If input_var2 is not supplied, this
            arguments is ignored. The indexing works like standard python list
            indexing. Index of 0 refers to the input layer (input_var) while
            an index of -1 points to the last hidden layer. Default parameter
            points to second layer from the end. If the model has only one
            layer, input_var2 is concatenated with that layer.
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
    n_layers = len(hidden_sizes) + 1
    _merge_inputs = False

    if input_var2 is not None:
        _merge_inputs = True
        if n_layers > 1:
            _concat_layer = (concat_layer % n_layers + n_layers) % n_layers
        else:
            _concat_layer = 0

    with tf.compat.v1.variable_scope(name):
        l_hid = input_var
        for idx, hidden_size in enumerate(hidden_sizes):
            if _merge_inputs and idx == _concat_layer:
                l_hid = tf.keras.layers.concatenate([l_hid, input_var2])

            l_hid = tf.layers.dense(inputs=l_hid,
                                    units=hidden_size,
                                    activation=hidden_nonlinearity,
                                    kernel_initializer=hidden_w_init,
                                    bias_initializer=hidden_b_init,
                                    name='hidden_{}'.format(idx))
            if layer_normalization:
                l_hid = tf.contrib.layers.layer_norm(l_hid)

        if _merge_inputs and _concat_layer == len(hidden_sizes):
            l_hid = tf.keras.layers.concatenate([l_hid, input_var2])

        l_out = tf.layers.dense(inputs=l_hid,
                                units=output_dim,
                                activation=output_nonlinearity,
                                kernel_initializer=output_w_init,
                                bias_initializer=output_b_init,
                                name='output')
    return l_out
