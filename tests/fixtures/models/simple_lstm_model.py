import tensorflow as tf

from garage.tf.models import Model


class SimpleLSTMModel(Model):
    """Simple LSTMModel for testing."""

    def __init__(self,
                 output_dim,
                 hidden_dim,
                 name='SimpleLSTMModel',
                 *args,
                 **kwargs):
        super().__init__(name)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def network_output_spec(self):
        """Network output spec."""
        return [
            'all_output', 'step_output', 'step_hidden', 'step_cell',
            'init_hidden', 'init_cell'
        ]

    def _build(self,
               obs_input,
               step_obs_input,
               step_hidden,
               step_cell,
               name=None):
        return_var = tf.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        outputs = tf.fill(
            (tf.shape(obs_input)[0], tf.shape(obs_input)[1], self.output_dim),
            return_var)
        output = tf.fill((tf.shape(step_obs_input)[0], self.output_dim),
                         return_var)

        hidden_init_var = tf.get_variable(
            name='initial_hidden',
            shape=(self.hidden_dim, ),
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=tf.float32)
        cell_init_var = tf.get_variable(
            name='initial_cell',
            shape=(self.hidden_dim, ),
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=tf.float32)

        return (outputs, output, step_hidden, step_cell, hidden_init_var,
                cell_init_var)
