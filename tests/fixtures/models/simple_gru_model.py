import tensorflow as tf

from garage.tf.models import Model


class SimpleGRUModel(Model):
    """Simple GRUModel for testing."""

    def __init__(self,
                 output_dim,
                 hidden_dim,
                 name='SimpleGRUModel',
                 *args,
                 **kwargs):
        super().__init__(name)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def network_input_spec(self):
        """Network input spec."""
        return ['full_input', 'step_input', 'step_hidden_input']

    def network_output_spec(self):
        """Network output spec."""
        return ['all_output', 'step_output', 'step_hidden', 'init_hidden']

    def _build(self, obs_input, step_obs_input, step_hidden, name=None):
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        outputs = tf.fill(
            (tf.shape(obs_input)[0], tf.shape(obs_input)[1], self.output_dim),
            return_var)
        output = tf.fill((tf.shape(step_obs_input)[0], self.output_dim),
                         return_var)

        hidden_init_var = tf.compat.v1.get_variable(
            name='initial_hidden',
            shape=(self.hidden_dim, ),
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=tf.float32)

        return outputs, output, step_hidden, hidden_init_var
