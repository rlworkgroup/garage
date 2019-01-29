import numpy as np


class DummyDistribution:
    def entropy(self, dist_info):
        return np.random.randn()
