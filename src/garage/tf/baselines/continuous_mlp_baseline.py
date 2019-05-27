"""This module implements continuous mlp baseline."""
import numpy as np

from garage.core import Serializable
from garage.misc.overrides import overrides
from garage.np.baselines import Baseline
from garage.tf.core import Parameterized
from garage.tf.regressors import ContinuousMLPRegressor


class ContinuousMLPBaseline(Baseline, Parameterized, Serializable):
    """A value function using a mlp network."""

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            regressor_args=None,
            name='ContinuousMLPBaseline',
    ):
        """
        Constructor.

        :param env_spec: environment specification.
        :param subsample_factor:
        :param num_seq_inputs: number of sequence inputs.
        :param regressor_args: regressor arguments.
        """
        Parameterized.__init__(self)
        Serializable.quick_init(self, locals())
        super(ContinuousMLPBaseline, self).__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = ContinuousMLPRegressor(
            input_shape=(
                env_spec.observation_space.flat_dim * num_seq_inputs, ),
            output_dim=1,
            name=name,
            **regressor_args)
        self.name = name

    @overrides
    def get_param_values(self, **tags):
        """Get parameter values."""
        return self._regressor.get_param_values(**tags)

    @overrides
    def set_param_values(self, val, **tags):
        """Set parameter values to val."""
        self._regressor.set_param_values(val, **tags)

    @overrides
    def fit(self, paths):
        """Fit regressor based on paths."""
        observations = np.concatenate([p['observations'] for p in paths])
        returns = np.concatenate([p['returns'] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    @overrides
    def predict(self, path):
        """Predict value based on paths."""
        return self._regressor.predict(path['observations']).flatten()

    @overrides
    def get_params_internal(self, **tags):
        return self._regressor.get_params_internal(**tags)
