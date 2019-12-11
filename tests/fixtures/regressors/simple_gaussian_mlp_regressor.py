"""Simple GaussianMLPRegressor for testing."""
import numpy as np
import tensorflow as tf

from garage.tf.regressors import StochasticRegressor
from tests.fixtures.models import SimpleGaussianMLPModel


class SimpleGaussianMLPRegressor(StochasticRegressor):
    """Simple GaussianMLPRegressor for testing.

    Args:
        input_shape (tuple[int]): Input shape of the training data.
        output_dim (int): Output dimension of the model.
        name (str): Model name, also the variable scope.
        args (list): Unused positionl arguments.
        kwargs (dict): Unused keyword arguments.

    """

    def __init__(self, input_shape, output_dim, name, *args, **kwargs):
        super().__init__(input_shape, output_dim, name)
        del args, kwargs
        self.model = SimpleGaussianMLPModel(output_dim=self._output_dim)

        self._ys = None
        self._initialize()

    @property
    def recurrent(self):
        """bool: If this module has a hidden state."""
        return False

    @property
    def vectorized(self):
        """bool: If this module supports vectorization input."""
        return True

    @property
    def distribution(self):
        """garage.tf.distributions.DiagonalGaussian: Distribution."""
        return self.model.networks['default'].dist

    def dist_info_sym(self, input_var, state_info_vars=None, name='default'):
        """Create a symbolic graph of the distribution parameters.

        Args:
            input_var (tf.Tensor): tf.Tensor of the input data.
            state_info_vars (dict): a dictionary whose values should contain
                information about the state of the policy at the time it
                received the input.
            name (str): Name of the new graph.

        Return:
            dict[tf.Tensor]: Outputs of the symbolic distribution parameter
                graph.

        """
        with tf.compat.v1.variable_scope(self._variable_scope):
            self.model.build(input_var, name=name)

        means_var = self.model.networks[name].means
        log_stds_var = self.model.networks[name].log_stds

        return dict(mean=means_var, log_std=log_stds_var)

    def _initialize(self):
        """Initialize graph."""
        input_ph = tf.compat.v1.placeholder(tf.float32,
                                            shape=(None, ) + self._input_shape)
        with tf.compat.v1.variable_scope(self._name) as vs:
            self._variable_scope = vs
            self.model.build(input_ph)

    def fit(self, xs, ys):
        """Fit with input data xs and label ys.

        Args:
            xs (numpy.ndarray): Input data.
            ys (numpy.ndarray): Label of input data.

        """
        self._ys = ys

    def predict(self, xs):
        """Predict ys based on input xs.

        Args:
            xs (numpy.ndarray): Input data.

        Return:
            np.ndarray: The predicted ys.

        """
        if self._ys is None:
            mean = tf.compat.v1.get_default_session().run(
                self.model.networks['default'].mean,
                feed_dict={self.model.networks['default'].input: xs})
            self._ys = np.full((len(xs), 1), mean)

        return self._ys

    def get_params_internal(self):
        """Get the params, which are the trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
            variable scope.

        """
        return self._variable_scope.trainable_variables()

    def __setstate__(self, state):
        """Object.__setstate__.

        Args:
            state (dict): Unpickled state.

        """
        super().__setstate__(state)
        self._initialize()
