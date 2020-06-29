"""Simple GaussianMLPModel for testing."""
import numpy as np
import tensorflow as tf

from garage.tf.distributions import DiagonalGaussian
from garage.tf.models import Model


class SimpleGaussianMLPModel(Model):
    """Simple SimpleGaussianMLPModel for testing.

    Args:
        output_dim (int): Dimension of the network output.
        name (str): Model name, also the variable scope.
        args (list): Unused positionl arguments.
        kwargs (dict): Unused keyword arguments.

    """

    # pylint: disable=arguments-differ
    def __init__(self,
                 output_dim,
                 *args,
                 name='SimpleGaussianMLPModel',
                 **kwargs):
        del args
        del kwargs
        super().__init__(name)
        self.output_dim = output_dim

    def network_output_spec(self):
        """Network output spec.

        Return:
            list[str]: List of key(str) for the network outputs.

        """
        return ['mean', 'log_std', 'dist']

    def _build(self, obs_input, name=None):
        """Build model.

        Args:
            obs_input (tf.Tensor): Entire time-series observation input.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Returns:
            tf.tensor: Mean.
            tf.Tensor: Log of standard deviation.
            garage.distributions.DiagonalGaussian: Distribution.

        """
        del name
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        mean = tf.fill((tf.shape(obs_input)[0], self.output_dim), return_var)
        log_std = tf.fill((tf.shape(obs_input)[0], self.output_dim),
                          np.log(0.5))
        dist = DiagonalGaussian(self.output_dim)
        # action will be 0.5 + 0.5 * 0.5 = 0.75
        return mean, log_std, dist
