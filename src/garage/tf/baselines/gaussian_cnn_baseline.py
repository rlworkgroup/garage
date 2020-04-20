"""Gaussian CNN Baseline."""
import akro
import numpy as np

from garage.misc.tensor_utils import normalize_pixel_batch
from garage.np.baselines import Baseline
from garage.tf.regressors import GaussianCNNRegressor


class GaussianCNNBaseline(Baseline):
    """GaussianCNNBaseline With Model.

    It fits the input data to a gaussian distribution estimated by a CNN.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        subsample_factor (float): The factor to subsample the data. By
            default it is 1.0, which means using all the data.
        regressor_args (dict): Arguments for regressor.
        name (str): Name of baseline.

    """

    def __init__(
            self,
            env_spec,
            subsample_factor=1.,
            regressor_args=None,
            name='GaussianCNNBaseline',
    ):
        if not isinstance(env_spec.observation_space, akro.Box) or \
                not len(env_spec.observation_space.shape) in (2, 3):
            raise ValueError(
                '{} can only process 2D, 3D akro.Image or'
                ' akro.Box observations, but received an env_spec with '
                'observation_space of type {} and shape {}'.format(
                    type(self).__name__,
                    type(env_spec.observation_space).__name__,
                    env_spec.observation_space.shape))

        super().__init__(env_spec)
        if regressor_args is None:
            regressor_args = dict()

        self._regressor = GaussianCNNRegressor(
            input_shape=(env_spec.observation_space.shape),
            output_dim=1,
            subsample_factor=subsample_factor,
            name=name,
            **regressor_args)
        self.name = name
        self.env_spec = env_spec

    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (dict[numpy.ndarray]): Sample paths.

        """
        observations = np.concatenate([p['observations'] for p in paths])
        if isinstance(self.env_spec.observation_space, akro.Image):
            observations = normalize_pixel_batch(observations)

        returns = np.concatenate([p['returns'] for p in paths])
        self._regressor.fit(observations, returns.reshape((-1, 1)))

    def predict(self, path):
        """Predict value based on paths.

        Args:
            path (dict[numpy.ndarray]): Sample paths.

        Returns:
            numpy.ndarray: Predicted value.

        """
        observations = path['observations']
        if isinstance(self.env_spec.observation_space, akro.Image):
            observations = normalize_pixel_batch(observations)

        return self._regressor.predict(observations).flatten()

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
