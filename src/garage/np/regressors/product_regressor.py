"""Product regressor."""
import numpy as np


class ProductRegressor:
    """Product regressor.

    A class for performing MLE regression by fitting a product distribution to
    the outputs. A separate regressor will be trained for each individual input
    distribution.

    Args:
        regressors (list[garage.tf.Regressor]): List of individual regressors

    """

    def __init__(self, regressors):
        self.regressors = regressors
        self.output_dims = [x.output_dim for x in regressors]

    def _split_ys(self, ys):
        """Split input label according to output dimension.

        Args:
            ys (numpy.ndarray): Label of input data.

        Returns:
            numpy.ndarray: Split labels.

        """
        ys = np.asarray(ys)
        split_ids = np.cumsum(self.output_dims)[:-1]
        return np.split(ys, split_ids, axis=1)

    def fit(self, xs, ys):
        """Fit with input data xs and label ys.

        Args:
            xs (numpy.ndarray): Input data.
            ys (numpy.ndarray): Label of input data.

        """
        for regressor, split_ys in zip(self.regressors, self._split_ys(ys)):
            regressor.fit(xs, split_ys)

    def predict(self, xs):
        """Predict ys based on input xs.

        Args:
            xs (numpy.ndarray): Input data.

        Return:
            np.ndarray: The predicted ys.

        """
        return np.concatenate(
            [regressor.predict(xs) for regressor in self.regressors], axis=1)

    def sample_predict(self, xs):
        """Sampling given input xs.

        Args:
            xs (numpy.ndarray): Input data.

        Returns:
            numpy.ndarray: The stochastic sampled ys.

        """
        return np.concatenate(
            [regressor.sample_predict(xs) for regressor in self.regressors],
            axis=1)

    def predict_log_likelihood(self, xs, ys):
        """Predict log-likelihood of output data conditioned on the input data.

        Args:
            xs (numpy.ndarray): Input data.
            ys (numpy.ndarray): Output labels in one hot representation.

        Return:
            numpy.ndarray: The predicted log likelihoods.

        """
        return np.sum([
         regressor.predict_log_likelihood(xs, split_ys)
         for regressor, split_ys in zip(self.regressors, self._split_ys(ys))
        ], axis=0)  # yapf: disable

    def get_param_values(self):
        """Return values of params.

        Returns:
            np.ndarray: Policy parameters values.

        """
        return np.concatenate(
            [regressor.get_param_values() for regressor in self.regressors])

    def set_param_values(self, flattened_params):
        """Set param values.

        Args:
            flattened_params (np.ndarray): Flattened parameter values.

        """
        param_dims = [
            np.prod(regressor.get_param_shapes())
            for regressor in self.regressors
        ]
        split_ids = np.cumsum(param_dims)[:-1]
        for regressor, split_param_values in zip(
                self.regressors, np.split(flattened_params, split_ids)):
            regressor.set_param_values(split_param_values)
