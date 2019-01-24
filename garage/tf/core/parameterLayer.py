"""Parameter layer in TensorFlow."""

import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import broadcast_to
from tensorflow.keras.layers import Layer

# flake8: noqa


class ParameterLayer(Layer):
    def __init__(self,
                 length,
                 initializer=tf.ones_initializer(),
                 dtype=tf.float32,
                 trainable=True,
                 name="parameter"):
        self._length = length
        self._initializer = initializer
        self._trainable = trainable
        super().__init__()

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(self._length, ),
            initializer=self._initializer,
            trainable=self._trainable)
        super().build(input_shape)

    def call(self, x):
        broadcast_shape = tf.concat(
            axis=0, values=[tf.shape(x)[:-1], [self._length]])
        return broadcast_to(self.kernel, shape=broadcast_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self._length)
