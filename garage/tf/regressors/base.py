"""Regressor base classes with Parameterized."""
import abc

from garage.tf.core import Parameterized


class Regressor(Parameterized, abc.ABC):
    """
    Regressor base class.

    Args:
        input_shape: Input shape.
        output_dim: Output dimension.
        name: Name of the regressor.
    """

    def __init__(self, input_shape, output_dim, name):
        Parameterized.__init__(self)
        self._input_shape = input_shape
        self._output_dim = output_dim
        self._name = name
        self._variable_scope = None

    def fit(self, xs, ys):
        """
        Fit with input data xs and label ys.

        Args:
            xs: Input data.
            ys: Label of input data.
        """
        raise NotImplementedError

    def predict(self, xs):
        """
        Predict ys based on input xs.

        Args:
            xs: Input data.
        """
        raise NotImplementedError

    def get_params_internal(self, **tags):
        """Get the list of parameters."""
        return self._variable_scope.trainable_variables()


class StochasticRegressor(Regressor):
    """
    StochasticRegressor base class.

    Args:
        input_shape: Input shape.
        output_dim: Output dimension.
        name: Name of the regressor.
    """

    def __init__(self, input_shape, output_dim, name):
        super().__init__(input_shape, output_dim, name)

    def log_likelihood_sym(self, x_var, y_var, name=None):
        """
        Symbolic graph of the log likelihood.

        Args:
            x_var: Input tf.Tensor for the input data.
            y_var: Input tf.Tensor for the label of data.
            name: Name of the new graph.
        """
        raise NotImplementedError

    def dist_info_sym(self, x_var, name=None):
        """
        Symbolic graph of the distribution.

        Args:
            x_var: Input tf.Tensor for the input data.
            name: Name of the new graph.
        """
        raise NotImplementedError
