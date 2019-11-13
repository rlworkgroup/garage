"""Gaussian CNN Baseline."""

import numpy as np

from garage.np.baselines import Baseline
from garage.tf.regressors import GaussianCNNRegressor


class GaussianCNNBaseline(Baseline):
    """
    GaussianCNNBaseline With Model.

    It fits the input data to a gaussian distribution estimated by a CNN.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        subsample_factor (float): The factor to subsample the data. By
            default it is 1.0, which means using all the data.
        regressor_args (dict): Arguments for regressor.
    """

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            regressor_args=None,
            name='GaussianCNNBaseline',
    ):
        super().__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianCNNRegressor(
            input_shape=(env_spec.observation_space.shape),
            output_dim=1,
            name=name,
            **regressor_args)
        self.name = name

    def fit(self, paths):
        """Fit regressor based on paths."""
        observations = np.concatenate([p['observations'] for p in paths])
        returns = np.concatenate([p['returns'] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    def predict(self, path):
        """Predict value based on paths."""
        return self._regressor.predict(path['observations']).flatten()

    def get_param_values(self, **tags):
        """Get parameter values."""
        return self._regressor.get_param_values(**tags)

    def set_param_values(self, flattened_params, **tags):
        """Set parameter values to val."""
        self._regressor.set_param_values(flattened_params, **tags)

    def get_params_internal(self, **tags):
        """Get parameter values."""
        return self._regressor.get_params_internal(**tags)
