"""
Multilayer Perceptrons (MLP) with tensorflow as the only dependency.

The module contains MLP which serves as the base of all networks.
It aims to replace existing implementation of MLP class
(garage.tf.core.network), which is under development.
"""
import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import broadcast_to


def mlp(input_var,
        output_dim,
        hidden_sizes,
        name,
        hidden_nonlinearity=tf.nn.relu,
        hidden_w_init=tf.contrib.layers.xavier_initializer(),
        hidden_b_init=tf.zeros_initializer(),
        output_nonlinearity=None,
        output_w_init=tf.contrib.layers.xavier_initializer(),
        output_b_init=tf.zeros_initializer(),
        layer_normalization=False):
    """
    MLP function.

    Args:
        input_var: Input tf.Tensor to the MLP.
        output_dim: Dimension of the network output.
        hidden_sizes: Output dimension of dense layer(s).
        name: variable scope of the mlp.
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
        layer_normalization: Bool for using layer normalization or not.

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
                name="hidden_{}".format(idx))
            if layer_normalization:
                l_hid = tf.contrib.layers.layer_norm(l_hid)
        l_out = tf.layers.dense(
            inputs=l_hid,
            units=output_dim,
            activation=output_nonlinearity,
            kernel_initializer=output_w_init,
            bias_initializer=output_b_init,
            name="output")
    return l_out


def parameter(input_var,
              length,
              initializer=tf.zeros_initializer(),
              dtype=tf.float32,
              trainable=True,
              name="parameter"):
    """
    Paramter function that creates  variables that could be broadcast to a certain shape to match with input var.
        
    Args:
        input_var: Input tf.Tensor.
        lenth: Integer dimension of the variables.
        initializer: Initializer of the variables.
        dtype: Data type of the variables.
        trainable: Whether these variables are trainable.
        name: variable scope of the variables.
    Return:
        A tensor of broadcasted variables
    """
    with tf.variable_scope(name):
        p = tf.get_variable(
            "parameter",
            shape=(length, ),
            dtype=dtype,
            initializer=initializer,
            trainable=trainable)

        ndim = input_var.get_shape().ndims
        broadcast_shape = tf.concat(
            axis=0, values=[tf.shape(input_var)[:ndim - 1], [length]])
        p_broadcast = broadcast_to(p, shape=broadcast_shape)
        return p_broadcast
