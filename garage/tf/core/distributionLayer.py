"""Parameter layer in TensorFlow."""
from tensorflow.keras.layers import Layer

# flake8: noqa


class DistributionLayer(Layer):
    def __init__(self, dist, trainable=True):
        self._dist = dist
        super().__init__()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, x):
        return self._dist(x[0], x[1]).sample()
