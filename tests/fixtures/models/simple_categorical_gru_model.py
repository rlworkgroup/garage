"""Simple CategoricalGRUModel for testing."""
import tensorflow_probability as tfp

from tests.fixtures.models.simple_gru_model import SimpleGRUModel


class SimpleCategoricalGRUModel(SimpleGRUModel):
    """Simple CategoricalGRUModel for testing.

    Args:
        output_dim (int): Dimension of the network output.
        hidden_dim (int): Hidden dimension for GRU cell.
        name (str): Policy name, also the variable scope.
        args: Extra arguments which are not used.
        kwargs: Extra keyword arguments which are not used.

    """

    def __init__(self, output_dim, hidden_dim, name, *args, **kwargs):
        super().__init__(output_dim, hidden_dim, name)

    def network_output_spec(self):
        """Network output spec.

        Returns:
            list[str]: Name of the model outputs, in order.

        """
        return [
            'all_output', 'step_output', 'step_hidden', 'init_hidden', 'dist'
        ]

    def _build(self, obs_input, step_obs_input, step_hidden, name=None):
        """Build model.

        Args:
            obs_input (tf.Tensor): Entire time-series observation input.
            step_obs_input (tf.Tensor): Single timestep observation input.
            step_hidden (tf.Tensor): Hidden state for step.
            name (str): Name of the model, also the name scope.

        Returns:
            tf.Tensor: Entire time-series outputs.
            tf.Tensor: Step output.
            tf.Tensor: Step hidden state.
            tf.Tensor: Initial hidden state.
            tfp.distributions.OneHotCategorical: Distribution.

        """
        outputs, output, step_hidden, hidden_init_var = super()._build(
            obs_input, step_obs_input, step_hidden, name)
        dist = tfp.distributions.OneHotCategorical(outputs)
        return outputs, output, step_hidden, hidden_init_var, dist
