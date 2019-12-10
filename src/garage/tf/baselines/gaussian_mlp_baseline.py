"""A value function (baseline) based on a GaussianMLP model."""
import numpy as np

from garage.np.baselines import Baseline
from garage.tf.regressors import GaussianMLPRegressor


class GaussianMLPBaseline(Baseline):
    """Gaussian MLP Baseline with Model.

    It fits the input data to a gaussian distribution estimated by
    a MLP.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        subsample_factor (float): The factor to subsample the data. By
            default it is 1.0, which means using all the data.
        num_seq_inputs (float): Number of sequence per input. By default
            it is 1.0, which means only one single sequence.
        regressor_args (dict): Arguments for regressor.
        name (str): Name of baseline.

    """

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            num_seq_inputs=1,
            regressor_args=None,
            name='GaussianMLPBaseline',
    ):
        super().__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianMLPRegressor(
            input_shape=(env_spec.observation_space.flat_dim *
                         num_seq_inputs, ),
            output_dim=1,
            name=name,
            subsample_factor=subsample_factor,
            **regressor_args)
        self.name = name

    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (list[dict]): Sample paths.

        """
        observations = np.concatenate([p['observations'] for p in paths])
        returns = np.concatenate([p['returns'] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    def predict(self, path):
        """Predict value based on paths.

        Args:
            path (list[dict]): Sample paths.

        Returns:
            numpy.ndarray: Predicted value.

        """
        return self._regressor.predict(path['observations']).flatten()

    def get_param_values(self):
        """Get parameter values.

        Returns:
            List[np.ndarray]: A list of values of each parameter.

        """
        return self._regressor.get_param_values()

    def set_param_values(self, flattened_params):
        """Set param values.

        Args:
            flattened_params (np.ndarray): A numpy array of parameter values.

        """
        self._regressor.set_param_values(flattened_params)

    def get_params_internal(self):
        """Get the params, which are the trainable variables.

        Returns:
            List[tf.Variable]: A list of trainable variables in the current
            variable scope.

        """
        return self._regressor.get_params_internal()
