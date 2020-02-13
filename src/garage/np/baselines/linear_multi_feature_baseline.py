"""Linear Multi-Feature Baseline."""
import numpy as np

from garage.np.baselines import LinearFeatureBaseline


class LinearMultiFeatureBaseline(LinearFeatureBaseline):
    """A linear value function (baseline) based on features.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        reg_coeff (float): Regularization coefficient.
        features (list[str]): Name of features.
        name (str): Name of baseline.

    """

    def __init__(self,
                 env_spec,
                 features=None,
                 reg_coeff=1e-5,
                 name='LinearMultiFeatureBaseline'):
        super().__init__(env_spec, reg_coeff, name)
        features = features or ['observation']
        self._feature_names = features

    def _features(self, path):
        """Extract features from path.

        Args:
            path (list[dict]): Sample paths.

        Returns:
            numpy.ndarray: Extracted features.

        """
        features = [
            np.clip(path[feature_name], -10, 10)
            for feature_name in self._feature_names
        ]
        n = len(path['rewards'])
        return np.concatenate(sum([[f, f**2]
                                   for f in features], []) + [np.ones((n, 1))],
                              axis=1)
