"""Policies which use NumPy as a numerical backend."""

from garage.np.policies.fixed_policy import FixedPolicy
from garage.np.policies.policy import Policy
from garage.np.policies.scripted_policy import ScriptedPolicy
from garage.np.policies.uniform_random_policy import UniformRandomPolicy

__all__ = ['FixedPolicy', 'Policy', 'ScriptedPolicy', 'UniformRandomPolicy']
