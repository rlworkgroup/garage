"""CNN model in TensorFlow."""

import tensorflow as tf


def cnn(input_var,
        output_dim,
        filter_dims,
        num_filters,
        stride,
        name,
        padding="VALID",
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.contrib.layers.xavier_initializer(),
        hidden_b_init=tf.zeros_initializer(),
        output_nonlinearity=None,
        output_w_init=tf.contrib.layers.xavier_initializer(),
        output_b_init=tf.zeros_initializer()):
    """
    CNN model. Based on 'NHWC' data format: [batch, height, width, channel].

    Args:
        input_var: Input tf.Tensor to the CNN.
        output_dim: Dimension of the network output.
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        stride: The stride of the sliding window.
        name: Variable scope of the cnn.
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).

    Return:
        The output tf.Tensor of the CNN.
    """
    strides = [1, stride, stride, 1]

    with tf.variable_scope(name):
        h = input_var
        for index, (filter_dim, num_filter) in enumerate(
                zip(filter_dims, num_filters)):
            h = _conv(h, 'h{}'.format(index), filter_dim, num_filter, strides,
                      hidden_w_init, hidden_b_init, padding)
            if hidden_nonlinearity is not None:
                h = hidden_nonlinearity(h)

        # convert conv to dense
        dim = tf.reduce_prod(h.get_shape()[1:].as_list())
        h = tf.reshape(h, [-1, dim.eval()])
        h = tf.layers.dense(
            inputs=h,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="output")

        return h


def cnn_with_max_pooling(input_var,
                         output_dim,
                         filter_dims,
                         num_filters,
                         stride,
                         name,
                         pool_shape,
                         pool_stride,
                         padding="VALID",
                         hidden_nonlinearity=tf.nn.relu,
                         hidden_w_init=tf.contrib.layers.xavier_initializer(),
                         hidden_b_init=tf.zeros_initializer(),
                         output_nonlinearity=None,
                         output_w_init=tf.contrib.layers.xavier_initializer(),
                         output_b_init=tf.zeros_initializer()):
    """
    CNN model. Based on 'NHWC' data format: [batch, height, width, channel].

    Args:
        input_var: Input tf.Tensor to the CNN.
        output_dim: Dimension of the network output.
        filter_dims: Dimension of the filters.
        num_filters: Number of filters.
        stride: The stride of the sliding window.
        name: Variable scope of the cnn.
        pool_shape: Dimension of the pooling layer(s).
        pool_stride: The stride of the pooling layer(s).
        padding: The type of padding algorithm to use, from "SAME", "VALID".
        hidden_nonlinearity: Activation function for
                    intermediate dense layer(s).
        hidden_w_init: Initializer function for the weight
                    of intermediate dense layer(s).
        hidden_b_init: Initializer function for the bias
                    of intermediate dense layer(s).
        output_nonlinearity: Activation function for
                    output dense layer.
        output_w_init: Initializer function for the weight
                    of output dense layer(s).
        output_b_init: Initializer function for the bias
                    of output dense layer(s).

    Return:
        The output tf.Tensor of the CNN.
    """
    strides = [1, stride, stride, 1]
    pool_strides = [1, pool_stride[0], pool_stride[1], 1]
    pool_shapes = [1, pool_shape[0], pool_shape[1], 1]

    with tf.variable_scope(name):
        h = input_var
        for index, (filter_dim, num_filter) in enumerate(
                zip(filter_dims, num_filters)):
            h = _conv(h, 'h{}'.format(index), filter_dim, num_filter, strides,
                      hidden_w_init, hidden_b_init, padding)
            if hidden_nonlinearity is not None:
                h = hidden_nonlinearity(h)
            h = tf.nn.max_pool(
                h, ksize=pool_shapes, strides=pool_strides, padding=padding)

        # convert conv to densevfxz
        dim = tf.reduce_prod(h.get_shape()[1:].as_list())
        h = tf.reshape(h, [-1, dim.eval()])
        h = tf.layers.dense(
            inputs=h,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="output")

        return h


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
