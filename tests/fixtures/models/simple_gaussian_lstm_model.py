import tensorflow as tf

from garage.tf.distributions import DiagonalGaussian
from garage.tf.models import Model


class SimpleGaussianLSTMModel(Model):
    """Simple GaussianLSTMModel for testing."""

    def __init__(self,
                 output_dim,
                 hidden_dim,
                 name='SimpleGaussianLSTMModel',
                 *args,
                 **kwargs):
        super().__init__(name)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def network_input_spec(self):
        """Network input spec."""
        return [
            'full_input', 'step_input', 'step_hidden_input', 'step_cell_input'
        ]

    def network_output_spec(self):
        """Network output spec."""
        return [
            'mean', 'step_mean', 'log_std', 'step_log_std', 'step_hidden',
            'step_cell', 'init_hidden', 'init_cell', 'dist'
        ]

    def _build(self,
               obs_input,
               step_obs_input,
               step_hidden,
               step_cell,
               name=None):
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        mean = log_std = tf.fill(
            (tf.shape(obs_input)[0], tf.shape(obs_input)[1], self.output_dim),
            return_var)
        step_mean = step_log_std = tf.fill(
            (tf.shape(step_obs_input)[0], self.output_dim), return_var)

        hidden_init_var = tf.compat.v1.get_variable(
            name='initial_hidden',
            shape=(self.hidden_dim, ),
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=tf.float32)
        cell_init_var = tf.compat.v1.get_variable(
            name='initial_cell',
            shape=(self.hidden_dim, ),
            initializer=tf.zeros_initializer(),
            trainable=False,
            dtype=tf.float32)

        dist = DiagonalGaussian(self.output_dim)
        # sample = 0.5 * 0.5 + 0.5 = 0.75
        return (mean, step_mean, log_std, step_log_std, step_hidden, step_cell,
                hidden_init_var, cell_init_var, dist)
