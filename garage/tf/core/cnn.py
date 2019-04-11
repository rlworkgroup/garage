"""CNN model in TensorFlow."""

import tensorflow as tf


def cnn(input_var,
        filter_dims,
        num_filters,
        strides,
        name,
        padding,
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.glorot_uniform_initializer(),
        hidden_b_init=tf.zeros_initializer()):
    """
    CNN model. Based on 'NHWC' data format: [batch, height, width, channel].

    Args:
        input_var: Input tf.Tensor to the CNN.
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        strides: The stride of the sliding window.
        name: Variable scope of the cnn.
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).

    Return:
        The output tf.Tensor of the CNN.
    """
    if padding not in ['SAME', 'VALID']:
        raise ValueError("Invalid padding: {}.".format(padding))
    with tf.variable_scope(name):
        h = input_var
        for index, (filter_dim, num_filter, stride) in enumerate(
                zip(filter_dims, num_filters, strides)):
            _stride = [1, stride, stride, 1]
            h = _conv(h, 'h{}'.format(index), filter_dim, num_filter, _stride,
                      hidden_w_init, hidden_b_init, padding)
            if hidden_nonlinearity is not None:
                h = hidden_nonlinearity(h)
        # flatten
        dim = tf.reduce_prod(h.get_shape()[1:].as_list())
        return tf.reshape(h, [-1, dim])


def cnn_with_max_pooling(input_var,
                         filter_dims,
                         num_filters,
                         strides,
                         name,
                         pool_shapes,
                         pool_strides,
                         padding,
                         hidden_nonlinearity=tf.nn.relu,
                         hidden_w_init=tf.glorot_uniform_initializer(),
                         hidden_b_init=tf.zeros_initializer()):
    """
    CNN model. Based on 'NHWC' data format: [batch, height, width, channel].

    Args:
        input_var: Input tf.Tensor to the CNN.
        output_dim: Dimension of the network output.
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        strides: The strides of the sliding window.
        name: Variable scope of the cnn.
        pool_shapes: Dimension of the pooling layer(s).
        pool_strides: The strides of the pooling layer(s).
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).

    Return:
        The output tf.Tensor of the CNN.
    """
    if padding not in ['SAME', 'VALID']:
        raise ValueError("Invalid padding: {}.".format(padding))
    pool_strides = [1, pool_strides[0], pool_strides[1], 1]
    pool_shapes = [1, pool_shapes[0], pool_shapes[1], 1]

    with tf.variable_scope(name):
        h = input_var
        for index, (filter_dim, num_filter, stride) in enumerate(
                zip(filter_dims, num_filters, strides)):
            _stride = [1, stride, stride, 1]
            h = _conv(h, 'h{}'.format(index), filter_dim, num_filter, _stride,
                      hidden_w_init, hidden_b_init, padding)
            if hidden_nonlinearity is not None:
                h = hidden_nonlinearity(h)
            h = tf.nn.max_pool(
                h, ksize=pool_shapes, strides=pool_strides, padding=padding)

        # flatten
        dim = tf.reduce_prod(h.get_shape()[1:].as_list())
        return tf.reshape(h, [-1, dim])


def _conv(input_var,
          name,
          filter_size,
          num_filter,
          strides,
          hidden_w_init,
          hidden_b_init,
          padding="VALID"):

    # channel from input
    input_shape = input_var.get_shape()[-1].value
    # [filter_height, filter_width, in_channels, out_channels]
    w_shape = [filter_size, filter_size, input_shape, num_filter]
    b_shape = [1, 1, 1, num_filter]

    with tf.variable_scope(name):
        weight = tf.get_variable('weight', w_shape, initializer=hidden_w_init)
        bias = tf.get_variable('bias', b_shape, initializer=hidden_b_init)

        return tf.nn.conv2d(
            input_var, weight, strides=strides, padding=padding) + bias
