"""Dummy Policy for algo tests.."""
import numpy as np

from garage.np.policies import Policy
from tests.fixtures.distributions import DummyDistribution


class DummyPolicy(Policy):
    """Dummy Policy.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(
            self,
            env_spec,
    ):
        super().__init__(env_spec=env_spec)
        self._param = []
        self._param_values = np.random.uniform(-1, 1, 1000)

    def get_action(self, observation):
        """Get single action from this policy for the input observation.

        Args:
            observation (numpy.ndarray): Observation from environment.

        Returns:
            numpy.ndarray: Predicted action.
            dict: Distribution parameters.

        """
        return self.action_space.sample(), dict(dummy='dummy')

    def get_actions(self, observations):
        """Get multiple actions from this policy for the input observations.

        Args:
            observations (numpy.ndarray): Observations from environment.

        Returns:
            numpy.ndarray: Predicted actions.
            dict: Distribution parameters.

        """
        n = len(observations)
        action, action_info = self.get_action(None)
        return [action] * n, action_info

    def get_params_internal(self):
        """Return a list of policy internal params.

        Returns:
            list: Policy parameters.

        """

        return self._param

    def get_param_values(self):
        """Return values of params.

        Returns:
            np.ndarray: Policy parameters values.

        """
        return self._param_values

    @property
    def distribution(self):
        """Return the distribution.

        Returns:
            garage.distribution: Policy distribution.

        """
        return DummyDistribution()

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: True if vectorized.

        """
        return True


class DummyPolicyWithoutVectorized(DummyPolicy):
    """Dummy Policy without vectorized

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.

    """

    def __init__(self, env_spec):
        super().__init__(env_spec=env_spec)

    @property
    def vectorized(self):
        """Vectorized or not.

        Returns:
            bool: True if vectorized.

        """
        return False
