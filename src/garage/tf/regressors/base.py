"""Regressor base classes without Parameterized."""
import abc

import tensorflow as tf

from garage.misc.tensor_utils import flatten_tensors
from garage.misc.tensor_utils import unflatten_tensors


class Regressor(abc.ABC):
    """Regressor base class.

    Args:
        input_shape (tuple[int]): Input shape.
        output_dim (int): Output dimension.
        name (str): Name of the regressor.

    """

    def __init__(self, input_shape, output_dim, name):
        self._input_shape = input_shape
        self._output_dim = output_dim
        self._name = name
        self._variable_scope = None
        self._cached_params = None
        self._cached_param_shapes = None

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

    def get_params_internal(self):
        """Get the list of parameters.

        This internal method does not perform caching, and should
        be implemented by subclasses.

        Returns:
            list[tf.Variable]: A list of trainable variables.

        """

    # pylint: disable=assignment-from-no-return
    def get_params(self):
        """Get the list of trainable parameters.

        Returns:
            list[tf.Variable]: A list of trainable variables in the current
                variable scope.

        """
        if self._cached_params is None:
            self._cached_params = self.get_params_internal()
        return self._cached_params

    def get_param_shapes(self):
        """Get the list of shapes for the parameters.

        Returns:
            List[tuple[int]]: A list of shapes of each parameter.

        """
        if self._cached_param_shapes is None:
            params = self.get_params()
            param_values = tf.compat.v1.get_default_session().run(params)
            self._cached_param_shapes = [val.shape for val in param_values]
        return self._cached_param_shapes

    def get_param_values(self):
        """Get the list of values for the parameters.

        Returns:
            List[np.ndarray]: A list of values of each parameter.

        """
        params = self.get_params()
        param_values = tf.compat.v1.get_default_session().run(params)
        return flatten_tensors(param_values)

    def set_param_values(self, param_values):
        """Set param values.

        Args:
            param_values (np.ndarray): A numpy array of parameter values.

        """
        param_values = unflatten_tensors(param_values, self.get_param_shapes())
        for param, value in zip(self.get_params(), param_values):
            param.load(value)

    def flat_to_params(self, flattened_params):
        """Unflatten tensors according to their respective shapes.

        Args:
            flattened_params (np.ndarray): A numpy array of flattened params.

        Returns:
            list[np.ndarray]: A list of parameters reshaped to the
                shapes specified.

        """
        return unflatten_tensors(flattened_params, self.get_param_shapes())

    def __getstate__(self):
        """Object.__getstate__.

        Returns:
            dict: The state to be pickled for the instance.

        """
        new_dict = self.__dict__.copy()
        del new_dict['_cached_params']
        return new_dict

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        self._cached_params = None
        self.__dict__.update(state)


class StochasticRegressor(Regressor):
    """StochasticRegressor base class."""

    def log_likelihood_sym(self, x_var, y_var, name=None):
        """Symbolic graph of the log likelihood.

        Args:
            x_var (tf.Tensor): Input tf.Tensor for the input data.
            y_var (tf.Tensor): Input tf.Tensor for the label of data.
            name (str): Name of the new graph.

        Return:
            tf.Tensor output of the symbolic log likelihood.

        """

    def dist_info_sym(self, x_var, name=None):
        """Symbolic graph of the distribution.

        Args:
            x_var (tf.Tensor): Input tf.Tensor for the input data.
            name (str): Name of the new graph.

        Return:
            tf.Tensor output of the symbolic distribution.

        """
