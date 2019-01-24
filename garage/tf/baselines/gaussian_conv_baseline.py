"""Gaussian Conv Baseline."""
import numpy as np

from garage.baselines import Baseline
from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.tf.core import Parameterized
from garage.tf.regressors import GaussianConvRegressor


class GaussianConvBaseline(Baseline, Parameterized, Serializable):
    """
    A Convolutional net Baseline with Gaussian dist output.

    Args:
        env_spec: a TfEnv.spec, which contains env spaces info.

    """

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            regressor_args=None,
    ):
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())
        super(GaussianConvBaseline, self).__init__(env_spec)

        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianConvRegressor(
            input_shape=env_spec.observation_space.shape,
            output_dim=1,
            name="GaussianConvBaseline",
            **regressor_args)

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
        """Get parameter values."""
        return self._regressor.get_params_internal(**tags)
