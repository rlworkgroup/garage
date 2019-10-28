"""Parameter layer in TensorFlow."""

import tensorflow as tf


def parameter(input_var,
              length,
              initializer=tf.zeros_initializer(),
              dtype=tf.float32,
              trainable=True,
              name='parameter'):
    """Parameter layer.

    Used as layer that could be broadcast to a certain shape to
    match with input variable during training.

    For recurrent usage, use garage.tf.models.recurrent_parameter().

    Example: A trainable parameter variable with shape (2,), it needs to be
    broadcasted to (32, 2) when applied to a batch with size 32.

    Args:
        input_var (tf.Tensor): Input tf.Tensor.
        length (int): Integer dimension of the variable.
        initializer (callable): Initializer of the variable. The function
            should return a tf.Tensor.
        dtype: Data type of the variable (default is tf.float32).
        trainable (bool): Whether the variable is trainable.
        name (str): Variable scope of the variable.

    Return:
        A tensor of the broadcasted variables.
    """
    with tf.compat.v1.variable_scope(name):
        p = tf.compat.v1.get_variable('parameter',
                                      shape=(length, ),
                                      dtype=dtype,
                                      initializer=initializer,
                                      trainable=trainable)
        batch_dim = tf.shape(input_var)[0]
        broadcast_shape = tf.concat(axis=0, values=[[batch_dim], [length]])
        p_broadcast = tf.broadcast_to(p, shape=broadcast_shape)
        return p_broadcast


def recurrent_parameter(input_var,
                        step_input_var,
                        length,
                        initializer=tf.zeros_initializer(),
                        dtype=tf.float32,
                        trainable=True,
                        name='recurrent_parameter'):
    """Parameter layer for recurrent networks.

    Used as layer that could be broadcast to a certain shape to
    match with input variable during training.

    Example: A trainable parameter variable with shape (2,), it needs to be
    broadcasted to (32, 4, 2) when applied to a batch with size 32 and
    time-length 4.

    Args:
        input_var (tf.Tensor): Input tf.Tensor for full time-series inputs.
        step_input_var (tf.Tensor): Input tf.Tensor for step inputs.
        length (int): Integer dimension of the variable.
        initializer (callable): Initializer of the variable. The function
            should return a tf.Tensor.
        dtype: Data type of the variable (default is tf.float32).
        trainable (bool): Whether the variable is trainable.
        name (str): Variable scope of the variable.

    Return:
        A tensor of the two broadcasted variables: one for full time-series
            inputs, one for step inputs.
    """
    with tf.compat.v1.variable_scope(name):
        p = tf.compat.v1.get_variable('parameter',
                            shape=(length, ),
                            dtype=dtype,
                            initializer=initializer,
                            trainable=trainable)
        batch_dim = tf.shape(input_var)[:2]
        step_batch_dim = tf.shape(step_input_var)[:1]
        broadcast_shape = tf.concat(axis=0, values=[batch_dim, [length]])
        step_broadcast_shape = tf.concat(axis=0,
                                         values=[step_batch_dim, [length]])
        p_broadcast = tf.broadcast_to(p, shape=broadcast_shape)
        step_p_broadcast = tf.broadcast_to(p, shape=step_broadcast_shape)
        return p_broadcast, step_p_broadcast
