import tensorflow as tf

from garage.tf.distributions import DiagonalGaussian
from garage.tf.models import Model


class SimpleGaussianCNNModel(Model):
    """Simple GaussianCNNModel for testing."""

    def __init__(self,
                 output_dim,
                 name='SimpleGaussianCNNModel',
                 *args,
                 **kwargs):
        super().__init__(name)
        self.output_dim = output_dim

    def network_output_spec(self):
        return ['sample', 'mean', 'log_std', 'std_param', 'dist']

    def _build(self, obs_input, name=None):
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        mean = tf.fill((tf.shape(obs_input)[0], self.output_dim), return_var)
        log_std = tf.fill((tf.shape(obs_input)[0], self.output_dim), 0.5)
        action = mean + log_std * 0.5
        dist = DiagonalGaussian(self.output_dim)
        # action will be 0.5 + 0.5 * 0.5 = 0.75
        return action, mean, log_std, log_std, dist
