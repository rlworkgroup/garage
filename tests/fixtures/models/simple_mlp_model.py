import tensorflow as tf

from garage.tf.models import Model


class SimpleMLPModel(Model):
    """Simple MLPModel for testing."""

    def __init__(self, output_dim, name=None, *args, **kwargs):
        super().__init__(name)
        self.output_dim = output_dim

    def _build(self, obs_input, name=None):
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        return tf.fill((tf.shape(obs_input)[0], self.output_dim), return_var)
