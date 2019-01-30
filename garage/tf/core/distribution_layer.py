"""Parameter layer in TensorFlow."""
from tensorflow.keras.layers import Layer as KerasLayer


class DistributionLayer(KerasLayer):
    """
    Distribution layer based on tf.keras.layers.Layer.

    Args:
        dist: the distribution used in this layer, with type tf.distributions.
        seed: seed used for distribution sampling.
    """

    def __init__(self, dist, seed=0, **kwargs):
        kwargs['name'] = "distribution_layer"
        super().__init__(**kwargs)
        self._dist_callable = dist
        self._seed = seed

    def build(self, input_shape):
        """tf.keras.layers.Layer build."""
        super().build(input_shape)

    def call(self, input):
        """tf.keras.layers.Layer call."""
        self._dist = self._dist_callable(*input)
        return self._dist.sample(seed=self._seed)

    def get_config(self):
        """Cusomterized configuration for serialization."""
        config = {
            "dist": self._dist_callable,
            "seed": self._seed,
        }

        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

    @property
    def dist(self):
        """Distribution."""
        return self._dist
