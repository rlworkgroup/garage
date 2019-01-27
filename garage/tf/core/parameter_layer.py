"""Parameter layer in TensorFlow."""
import tensorflow as tf
from tensorflow.keras.layers import Layer as KerasLayer
from tensorflow.python.ops.gen_array_ops import broadcast_to


class ParameterLayer(KerasLayer):
    """
    Parameter layer based on tf.keras.layers.Layer.

    Used as layer that could be broadcast to a certain shape to
    match with input variable during training.
    Example: A trainable parameter variable with shape (2,), it needs to be
    broadcasted to (32, 2) when applied to a batch with size 32.

    Args:
        length: Size of the parameter variable.
        scope: Name scope of the parameter variable.
        initializer: Initializer for the parameter variable.
        trainable: If the parameter variable is trainable.
    """

    def __init__(self,
                 length,
                 scope="ParameterLayer",
                 initializer="ones",
                 trainable=True,
                 **kwargs):
        self.length = length
        self.initializer = initializer
        self.trainable = trainable
        self.scope = scope
        super().__init__(**kwargs)

    def build(self, input_shape):
        """tf.keras.layers.Layer build."""
        with tf.name_scope(self.scope):
            self.kernel = self.add_weight(
                name='kernel',
                shape=(self.length, ),
                initializer=self.initializer,
                trainable=self.trainable)
            super().build(input_shape)

    def call(self, x):
        """tf.keras.layers.Layer call."""
        broadcast_shape = tf.concat(
            axis=0, values=[tf.shape(x)[:-1], [self.length]])
        return broadcast_to(self.kernel, shape=broadcast_shape)

    def get_config(self):
        """Cusomterized configuration for serialization."""
        config = {
            'length': self.length,
            'scope': self.scope,
            'initializer': self.initializer,
            'trainable': self.trainable
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
