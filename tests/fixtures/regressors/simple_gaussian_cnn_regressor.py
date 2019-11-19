import numpy as np
import tensorflow as tf

from garage.tf.regressors import StochasticRegressor
from tests.fixtures.models import SimpleGaussianCNNModel


class SimpleGaussianCNNRegressor(StochasticRegressor):
    """Simple GaussianCNNRegressor for testing."""

    def __init__(self, input_shape, output_dim, name, *args, **kwargs):
        super().__init__(input_shape, output_dim, name)

        self.model = SimpleGaussianCNNModel(output_dim=self._output_dim)

        self._ys = None
        self._initialize()

    def _initialize(self):
        input_ph = tf.compat.v1.placeholder(tf.float32,
                                            shape=(None, ) + self._input_shape)
        with tf.compat.v1.variable_scope(self._name) as vs:
            self._variable_scope = vs
            self.model.build(input_ph)

    def fit(self, xs, ys):
        self._ys = ys

    def predict(self, xs):
        if self._ys is None:
            mean = tf.compat.v1.get_default_session().run(
                self.model.networks['default'].mean,
                feed_dict={self.model.networks['default'].input: xs})
            self._ys = np.full((len(xs), 1), mean)

        return self._ys

    def get_params_internal(self, *args, **kwargs):
        return self._variable_scope.trainable_variables()

    def __setstate__(self, state):
        """Object.__setstate__."""
        super().__setstate__(state)
        self._initialize()
