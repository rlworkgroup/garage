"""Simple MLPModel for testing."""
import tensorflow as tf

from garage.tf.models import Model


class SimpleMLPModel(Model):
    """Simple MLPModel for testing.

    Args:
        output_dim (int): Dimension of the network output.
        name (str): Model name, also the variable scope.
        args (list): Unused positionl arguments.
        kwargs (dict): Unused keyword arguments.

    """

    # pylint: disable=arguments-differ
    def __init__(self, output_dim, *args, name=None, **kwargs):
        del args
        del kwargs
        super().__init__(name)
        self.output_dim = output_dim

    def _build(self, obs_input, name=None):
        """Build model given input placeholder(s).

        Args:
            obs_input (tf.Tensor): Tensor input for state.
            name (str): Inner model name, also the variable scope of the
                inner model, if exist. One example is
                garage.tf.models.Sequential.

        Return:
            tf.Tensor: Tensor output of the model.

        """
        del name
        return_var = tf.compat.v1.get_variable(
            'return_var', (), initializer=tf.constant_initializer(0.5))
        return tf.fill((tf.shape(obs_input)[0], self.output_dim), return_var)
