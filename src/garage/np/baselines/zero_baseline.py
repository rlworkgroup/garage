import numpy as np

from garage.np.baselines.baseline import Baseline


class ZeroBaseline(Baseline):

    def __init__(self, env_spec):
        pass

    def get_param_values(self, **kwargs):
        return None

    def set_param_values(self, val, **kwargs):
        pass

    def fit(self, paths):
        pass

    def predict(self, path):
        return np.zeros_like(path['rewards'])

    def predict_n(self, paths):
        return [np.zeros_like(path['rewards']) for path in paths]
