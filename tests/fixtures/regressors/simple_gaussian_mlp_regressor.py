import tensorflow as tf

from garage.tf.regressors import StochasticRegressor


class SimpleGaussianMLPRegressor(StochasticRegressor):
    """Simple GaussianMLPModel for testing."""

    def __init__(self, input_shape, output_dim, name, *args, **kwargs):
        super().__init__(input_shape, output_dim, name)
        self.param_values = tf.get_variable("{}/params".format(name), [100])
        tf.get_default_session().run(
            tf.variables_initializer([self.param_values]))
        self.ys = None

    def fit(self, xs, ys):
        self.ys = ys

    def predict(self, xs):
        return self.ys

    def get_param_values(self, *args, **kwargs):
        return tf.get_default_session().run(self.param_values)

    def set_param_values(self, flattened_params, *args, **kwargs):
        tf.get_default_session().run(
            tf.assign(self.param_values, flattened_params))

    def get_params_internal(self, *args, **kwargs):
        return [self.param_values]
