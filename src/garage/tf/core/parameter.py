"""Parameter layer in TensorFlow."""

import tensorflow as tf


def parameter(length,
              initializer=tf.zeros_initializer(),
              dtype=tf.float32,
              trainable=True,
              name='parameter'):
    """
    Parameter layer.

    Helper function to create a tf.Variable under the scope name

    Args:
        length (int): Integer dimension of the variable.
        initializer (callable): Initializer of the variable. The function
            should return a tf.Tensor.
        dtype: Data type of the variable (default is tf.float32).
        trainable (bool): Whether the variable is trainable.
        name (str): Variable scope of the variable.

    Return:
        The tf.Tensor variable.
    """
    with tf.variable_scope(name):
        return tf.get_variable(
            'parameter',
            shape=(length, ),
            dtype=dtype,
            initializer=initializer,
            trainable=trainable)
