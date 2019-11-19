import numpy as np

from garage.np.baselines.base import Baseline


class LinearFeatureBaseline(Baseline):

    def __init__(self, env_spec, reg_coeff=1e-5, name='LinearFeatureBaseline'):
        self._coeffs = None
        self._reg_coeff = reg_coeff
        self.name = name

    def get_param_values(self, **tags):
        return self._coeffs

    def set_param_values(self, val, **tags):
        self._coeffs = val

    def _features(self, path):
        obs = np.clip(path['observations'], -10, 10)
        length = len(path['rewards'])
        al = np.arange(length).reshape(-1, 1) / 100.0
        return np.concatenate(
            [obs, obs**2, al, al**2, al**3,
             np.ones((length, 1))], axis=1)

    def fit(self, paths):
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

    def predict(self, path):
        if self._coeffs is None:
            return np.zeros(len(path['rewards']))
        return self._features(path).dot(self._coeffs)
