"""Linear Feature Baseline for Multitasks."""
import numpy as np

from garage.np.baselines import LinearFeatureBaseline


class MultiTaskLinearFeatureBaseline(LinearFeatureBaseline):
    def _features(self, path):
        t = np.clip(path['tasks'], -10, 10)
        o = np.clip(path['observations'], -10, 10)
        z = np.clip(path['latents'], -10, 10)
        n = len(path['rewards'])
        an = np.arange(n).reshape(-1, 1) / 100.0
        return np.concatenate(
            [t, t**2, o, o**2, z, z**2, an, an**2, an**3,
             np.ones((n, 1))],
            axis=1)
