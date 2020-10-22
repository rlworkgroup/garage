"""A linear value function (baseline) based on features."""
import numpy as np

from garage.np.baselines.baseline import Baseline


class LinearFeatureBaseline(Baseline):
    """A linear value function (baseline) based on features.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        reg_coeff (float): Regularization coefficient.
        name (str): Name of baseline.

    """

    def __init__(self, env_spec, reg_coeff=1e-5, name='LinearFeatureBaseline'):
        del env_spec
        self._coeffs = None
        self._reg_coeff = reg_coeff
        self.name = name
        self.lower_bound = -10
        self.upper_bound = 10

    def get_param_values(self):
        """Get parameter values.

        Returns:
            List[np.ndarray]: A list of values of each parameter.

        """
        return self._coeffs

    def set_param_values(self, flattened_params):
        """Set param values.

        Args:
            flattened_params (np.ndarray): A numpy array of parameter values.

        """
        self._coeffs = flattened_params

    def _features(self, path):
        """Extract features from path.

        Args:
            path (list[dict]): Sample paths.

        Returns:
            numpy.ndarray: Extracted features.

        """
        obs = np.clip(path['observations'], self.lower_bound, self.upper_bound)
        length = len(path['observations'])
        al = np.arange(length).reshape(-1, 1) / 100.0
        return np.concatenate(
            [obs, obs**2, al, al**2, al**3,
             np.ones((length, 1))], axis=1)

    # pylint: disable=unsubscriptable-object
    def fit(self, paths):
        """Fit regressor based on paths.

        Args:
            paths (list[dict]): Sample paths.

        """
        featmat = np.concatenate([self._features(path) for path in paths])
        returns = np.concatenate([path['returns'] for path in paths])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) +
                reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns),
                rcond=-1)[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def predict(self, paths):
        """Predict value based on paths.

        Args:
            paths (list[dict]): Sample paths.

        Returns:
            numpy.ndarray: Predicted value.

        """
        if self._coeffs is None:
            return np.zeros(len(paths['observations']))
        return self._features(paths).dot(self._coeffs)
