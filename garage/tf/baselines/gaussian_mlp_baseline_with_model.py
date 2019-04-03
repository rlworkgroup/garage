"""This module implements gaussian mlp baseline."""
import numpy as np

from garage.misc.overrides import overrides
from garage.np.baselines import Baseline
from garage.tf.regressors import GaussianMLPRegressorWithModel


class GaussianMLPBaselineWithModel(Baseline):
    """A value function using gaussian mlp network."""

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            regressor_args=None,
            name="GaussianMLPBaselineWithModel",
    ):
        """
        Constructor.

        :param env_spec:
        :param subsample_factor:
        :param num_seq_inputs:
        :param regressor_args:
        """
        super().__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLPRegressorWithModel(
            input_shape=(
                env_spec.observation_space.flat_dim * num_seq_inputs, ),
            output_dim=1,
            name=name,
            **regressor_args)
        self.name = name

    @overrides
    def fit(self, paths):
        """Fit regressor based on paths."""
        observations = np.concatenate([p["observations"] for p in paths])
        returns = np.concatenate([p["returns"] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        """Predict value based on paths."""
        return self._regressor.predict(path["observations"]).flatten()

    @overrides
    def get_param_values(self, **tags):
        """Get parameter values."""
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, flattened_params, **tags):
        """Set parameter values to val."""
        self._regressor.set_param_values(flattened_params, **tags)

    @overrides
    def get_params_internal(self, **tags):
        """Get internal parameters."""
        return self._regressor.get_params_internal(**tags)
