import numpy as np

from garage.misc.overrides import overrides
from garage.np.baselines.base import Baseline


class ZeroBaseline(Baseline):
    def __init__(self, env_spec):
        pass

    @overrides
    def get_param_values(self, **kwargs):
        return None

    @overrides
    def set_param_values(self, val, **kwargs):
        pass

    @overrides
    def fit(self, paths):
        pass

    @overrides
    def predict(self, path):
        return np.zeros_like(path['rewards'])

    @overrides
    def predict_n(self, paths):
        return [np.zeros_like(path['rewards']) for path in paths]
