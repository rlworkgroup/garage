"""CNN in TensorFlow."""

import tensorflow as tf

from garage.experiment import deterministic


def cnn(input_var,
        filters,
        strides,
        name,
        padding,
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.initializers.glorot_uniform(
            seed=deterministic.get_tf_seed_stream()),
        hidden_b_init=tf.zeros_initializer()):
    """Convolutional neural network (CNN).

    Note:
        Based on 'NHWC' data format: [batch, height, width, channel].

    Args:
        input_var (tf.Tensor): Input tf.Tensor to the CNN.
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        name (str): Network name, also the variable scope.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.

    Return:
        tf.Tensor: The output tf.Tensor of the CNN.

    """
    with tf.compat.v1.variable_scope(name):
        h = input_var
        for index, (filter_iter, stride) in enumerate(zip(filters, strides)):
            _stride = [1, stride, stride, 1]
            h = _conv(h, 'h{}'.format(index), filter_iter[1], filter_iter[0],
                      _stride, hidden_w_init, hidden_b_init, padding)
            if hidden_nonlinearity is not None:
                h = hidden_nonlinearity(h)

        # flatten
        dim = tf.reduce_prod(h.get_shape()[1:].as_list())
        return tf.reshape(h, [-1, dim])


def cnn_with_max_pooling(input_var,
                         filters,
                         strides,
                         name,
                         pool_shapes,
                         pool_strides,
                         padding,
                         hidden_nonlinearity=tf.nn.relu,
                         hidden_w_init=tf.initializers.glorot_uniform(
                             seed=deterministic.get_tf_seed_stream()),
                         hidden_b_init=tf.zeros_initializer()):
    """Convolutional neural network (CNN) with max-pooling.

    Note:
        Based on 'NHWC' data format: [batch, height, width, channel].

    Args:
        input_var (tf.Tensor): Input tf.Tensor to the CNN.
        filters (Tuple[Tuple[int, Tuple[int, int]], ...]): Number and dimension
            of filters. For example, ((3, (3, 5)), (32, (3, 3))) means there
            are two convolutional layers. The filter for the first layer have 3
            channels and its shape is (3 x 5), while the filter for the second
            layer have 32 channels and its shape is (3 x 3).
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        name (str): Model name, also the variable scope of the cnn.
        pool_shapes (tuple[int]): Dimension of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            shape (2, 2).
        pool_strides (tuple[int]): The strides of the pooling layer(s). For
            example, (2, 2) means that all the pooling layers have
            strides (2, 2).
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a tf.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.

    Return:
        tf.Tensor: The output tf.Tensor of the CNN.

    """
    pool_strides = [1, pool_strides[0], pool_strides[1], 1]
    pool_shapes = [1, pool_shapes[0], pool_shapes[1], 1]

    with tf.compat.v1.variable_scope(name):
        h = input_var
        for index, (filter_iter, stride) in enumerate(zip(filters, strides)):
            _stride = [1, stride, stride, 1]
            h = _conv(h, 'h{}'.format(index), filter_iter[1], filter_iter[0],
                      _stride, hidden_w_init, hidden_b_init, padding)
            if hidden_nonlinearity is not None:
                h = hidden_nonlinearity(h)
            h = tf.nn.max_pool2d(h,
                                 ksize=pool_shapes,
                                 strides=pool_strides,
                                 padding=padding)

        # flatten
        dim = tf.reduce_prod(h.get_shape()[1:].as_list())
        return tf.reshape(h, [-1, dim])


def _conv(input_var, name, filter_size, num_filter, strides, hidden_w_init,
          hidden_b_init, padding):
    """Helper function for performing convolution.

    Args:
        input_var (tf.Tensor): Input tf.Tensor to the CNN.
        name (str): Variable scope of the convolution Op.
        filter_size (tuple[int]): Dimension of the filter. For example,
            (3, 5) means the dimension of the filter is (3 x 5).
        num_filter (int): Number of filters. For example, (3, 32) means
            there are two convolutional layers. The filter for the first layer
            has 3 channels and the second one with 32 channels.
        strides (tuple[int]): The stride of the sliding window. For example,
            (1, 2) means there are two convolutional layers. The stride of the
            filter for first layer is 1 and that of the second layer is 2.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            tf.Tensor.
        padding (str): The type of padding algorithm to use,
            either 'SAME' or 'VALID'.

    Return:
        tf.Tensor: The output of the convolution.

    """
    # channel from input
    input_shape = input_var.get_shape()[-1]
    # [filter_height, filter_width, in_channels, out_channels]
    w_shape = [filter_size[0], filter_size[1], input_shape, num_filter]
    b_shape = [1, 1, 1, num_filter]

    with tf.compat.v1.variable_scope(name):
        weight = tf.compat.v1.get_variable('weight',
                                           w_shape,
                                           initializer=hidden_w_init)
        bias = tf.compat.v1.get_variable('bias',
                                         b_shape,
                                         initializer=hidden_b_init)

        return tf.nn.conv2d(
            input_var, weight, strides=strides, padding=padding) + bias
