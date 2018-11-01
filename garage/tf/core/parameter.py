"""Parameter variable in TensorFlow."""

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import broadcast_to


def parameter(input_var,
              length,
              initializer=tf.zeros_initializer(),
              dtype=tf.float32,
              trainable=True,
              name="parameter"):
    """
    Parameter variable.

    It creates variables that could be broadcast to a certain shape to
    match with input var.

    Args:
        input_var: Input tf.Tensor.
        length: Integer dimension of the variables.
        initializer: Initializer of the variables.
        dtype: Data type of the variables.
        trainable: Whether these variables are trainable.
        name: variable scope of the variables.

    Return:
        A tensor of broadcasted variables.
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
