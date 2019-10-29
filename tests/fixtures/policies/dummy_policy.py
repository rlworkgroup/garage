"""Dummy Policy for algo tests.."""
import numpy as np

from garage.np.policies import Policy
from tests.fixtures.distributions import DummyDistribution


class DummyPolicy(Policy):
    """Dummy Policy."""

    def __init__(
            self,
            env_spec,
    ):
        super().__init__(env_spec=env_spec)

    def get_action(self, observation):
        """Return action."""
        return self.action_space.sample(), dict()

    def get_actions(self, obses):
        """Return actions."""
        n = len(obses)
        action, action_info = self.get_action(None)
        return [action] * n, action_info

    def get_params_internal(self, **tags):
        """Return a list of policy internal params."""
        return []

    def get_param_values(self, **tags):
        """Return values of params."""
        return np.random.uniform(-1, 1, 1000)

    def reset(self, *args, **kwargs):
        pass

    @property
    def distribution(self):
        """Return the distribution."""
        return DummyDistribution()

    @property
    def vectorized(self):
        return True


class DummyPolicyWithoutVectorized(DummyPolicy):
    """Dummy Policy without vectorized"""

    def __init__(self, env_spec):
        super().__init__(env_spec=env_spec)

    @property
    def vectorized(self):
        return False
