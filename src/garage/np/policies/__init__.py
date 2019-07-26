"""Policies which use NumPy as a numerical backend."""
from garage.np.policies.base import Policy
from garage.np.policies.base import StochasticPolicy
from garage.np.policies.scripted_policy import ScriptedPolicy
__all__ = ['Policy', 'StochasticPolicy', 'ScriptedPolicy']
