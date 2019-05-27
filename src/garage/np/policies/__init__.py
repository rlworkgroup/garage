"""Policies which use NumPy as a numerical backend."""
from garage.np.policies.base import Policy
from garage.np.policies.base import StochasticPolicy

__all__ = ['Policy', 'StochasticPolicy']
