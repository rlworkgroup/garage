"""Simple CategoricalMLPModel for testing."""
import tensorflow_probability as tfp

from tests.fixtures.models.simple_mlp_model import SimpleMLPModel


class SimpleCategoricalMLPModel(SimpleMLPModel):
    """Simple CategoricalMLPModel for testing.

    Args:
        output_dim (int): Dimension of the network output.
        name (str): Policy name, also the variable scope.
        args: Extra arguments which are not used.
        kwargs: Extra keyword arguments which are not used.

    """

    def __init__(self, output_dim, name, *args, **kwargs):
        super().__init__(output_dim, name)

    def network_output_spec(self):
        """Network output spec.

        Returns:
            list[str]: Name of the model outputs, in order.

        """
        return ['prob', 'dist']

    def _build(self, obs_input, name=None):
        """Build model.

        Args:
            obs_input (tf.Tensor): Observation inputs.
            name (str): Name of the model, also the name scope.

        Returns:
            tf.Tensor: Network outputs.
            tfp.distributions.OneHotCategorical: Distribution.

        """
        prob = super()._build(obs_input, name)
        dist = tfp.distributions.OneHotCategorical(prob)
        return prob, dist
