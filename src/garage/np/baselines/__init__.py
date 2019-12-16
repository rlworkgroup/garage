"""Baselines (value functions) which use NumPy as a numerical backend."""
from garage.np.baselines.base import Baseline
from garage.np.baselines.linear_feature_baseline import LinearFeatureBaseline
from garage.np.baselines.zero_baseline import ZeroBaseline

__all__ = ['Baseline', 'LinearFeatureBaseline', 'ZeroBaseline']
