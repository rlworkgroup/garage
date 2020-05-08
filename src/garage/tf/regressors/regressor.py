"""Regressor base classes without Parameterized."""
from garage.tf.models import Module, StochasticModule


class Regressor(Module):
    """Regressor base class.

    Args:
        input_shape (tuple[int]): Input shape.
        output_dim (int): Output dimension.
        name (str): Name of the regressor.

    """

    # pylint: disable=abstract-method

    def __init__(self, input_shape, output_dim, name):
        super().__init__(name)
        self._input_shape = input_shape
        self._output_dim = output_dim

    def fit(self, xs, ys):
        """Fit with input data xs and label ys.

        Args:
            xs (numpy.ndarray): Input data.
            ys (numpy.ndarray): Label of input data.

        """

    def predict(self, xs):
        """Predict ys based on input xs.

        Args:
            xs (numpy.ndarray): Input data.

        Return:
            The predicted ys.

        """


class StochasticRegressor(Regressor, StochasticModule):
    """StochasticRegressor base class."""

    # pylint: disable=abstract-method

    def log_likelihood_sym(self, x_var, y_var, name=None):
        """Symbolic graph of the log likelihood.

        Args:
            x_var (tf.Tensor): Input tf.Tensor for the input data.
            y_var (tf.Tensor): Input tf.Tensor for the label of data.
            name (str): Name of the new graph.

        Return:
            tf.Tensor output of the symbolic log likelihood.

        """
