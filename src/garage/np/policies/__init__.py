"""Policies which use NumPy as a numerical backend."""

from garage.np.policies.fixed_policy import FixedPolicy
from garage.np.policies.policy import Policy
from garage.np.policies.scripted_policy import ScriptedPolicy

__all__ = [
    'FixedPolicy',
    'Policy',
    'ScriptedPolicy',
]
