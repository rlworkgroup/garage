import numpy as np

from garage.baselines import Baseline
from garage.misc.overrides import overrides


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
        return np.zeros_like(path["rewards"])

    @overrides
    def predict_n(self, paths):
        return [np.zeros_like(path["rewards"]) for path in paths]
